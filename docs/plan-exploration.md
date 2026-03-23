# Exploration & Validation Plan

Date: 2026-03-20
Status: Draft — ready for review before implementation
Supersedes: v1 (2026-03-19)

This document covers Stage 13 (data validation) and Stage 14 (exploratory analysis). It runs on the cleaned `unified.parquet` produced by the preprocessing pipeline (Stages 1-12 in `plan-preprocessing.md`). For hypothesis testing, see `plan-analysis.md`.

Research questions (from `1-research-design.md`):

- **RQ1:** Employer-side restructuring (junior share/volume, junior scope inflation, senior role redefinition, source/metro heterogeneity)
- **RQ2:** Task and requirement migration
- **RQ3:** Employer-requirement / worker-usage divergence (posting AI requirements vs. worker-side observed AI usage benchmarks such as Anthropic occupation-level data)
- **RQ4:** Mechanisms (interview-based, qualitative)

---

## Data sources

Three datasets, each with distinct strengths and gaps:

| | Kaggle (arshkon) | Kaggle (asaniczka) | Scraped (March 20+ 2026, new format only) |
|---|---|---|---|
| **Source** | `arshkon/linkedin-job-postings` | `asaniczka/1-3m-linkedin-jobs-and-skills-2024` | Our daily scraper |
| **Total rows** | 124K | 1.35M | ~3,680 SWE/day + ~30,888 non-SWE/day |
| **Date range** | April 5-20, 2024 | January 12-17, 2024 | March 20, 2026 onward (ongoing) |
| **Platform** | LinkedIn only | LinkedIn only | LinkedIn + Indeed |
| **SWE postings** | ~2,604 (~2.1%) | ~18,169 US SWE matches | ~3,680/day |
| **Descriptions** | Full text inline | Separate file (96.2% join rate) | Full text inline |
| **Seniority labels** | Yes (entry, mid-senior, etc.) | NO entry-level (only "Mid senior" and "Associate") | LinkedIn: job_level 100%; Indeed: 0% |
| **Company size/industry** | Via companion file joins | NO | LinkedIn: industry 100%; Indeed: company_size 91% |
| **Date posted** | Yes | Yes | LinkedIn: 2.8%; Indeed: 100% |
| **Search metadata** | None | None | search_query, query_tier, search_metro_name (41 columns) |

**Critical gap:** asaniczka has NO entry-level seniority labels. All entry-level analysis for the historical baseline must come from arshkon, which yields only ~385 entry-level SWE postings after filtering. This is the binding constraint for power in RQ1 junior-share analyses.

**Cross-dataset comparability** now involves THREE datasets (arshkon April 2024, asaniczka January 2024, scraped March 2026), not two. Validation must address both the arshkon-vs-asaniczka overlap period (both 2024, different months) and the 2024-vs-2026 temporal comparison.

**Primary analysis platform:** LinkedIn only. Indeed data is used for sensitivity analyses only.

---

## Stage 13: Data validation

This is the core validation battery. Every check here answers a question a reviewer would ask. Results feed into the methodology section and determine whether downstream analyses are defensible.

The validation stage runs AFTER preprocessing (Stages 1-12) is complete, on the cleaned unified dataset. Some checks may trigger iteration on earlier stages (e.g., if representativeness is poor, revisit dedup thresholds or geographic filtering).

### 13a. Representativeness: Do our scraped data look like the real labor market?

**Why a reviewer asks:** "Your data is whatever LinkedIn's algorithm gave you. How do you know it reflects actual job openings?"

**Benchmark sources:**
- **JOLTS** (via FRED): Total job openings, Professional & Business Services, Information sector. Free CSV download. Establishes macro-level plausibility.
- **BLS OES** (Occupational Employment & Wage Statistics): Occupation x metro employment counts. The gold standard for occupation-level validation.
- **Revelio Labs** (already ingested): SOC-15 aggregate hiring and openings trends.

**Tests to run:**

| Test | What it measures | Acceptable threshold | Reference |
|---|---|---|---|
| **Occupation share correlation** | Pearson r between our occupation distribution and OES employment shares | r > 0.80 (Hershbein & Kahn got 0.84-0.98) | Hershbein & Kahn (2018) |
| **Industry share comparison** | Our SWE industry distribution vs. OES SWE industry distribution | Within 5pp per industry | OECD (2024) |
| **Geographic share correlation** | Our state-level SWE counts vs. OES state-level SWE employment | r > 0.80 | Hershbein & Kahn (2018) |
| **Dissimilarity index** | Duncan index between our occupation distribution and OES | Report and discuss (no hard cutoff) | OECD (2024) |
| **Posting volume vs. JOLTS** | Time-series correlation between our daily posting counts and JOLTS information sector openings | r > 0.60 (Turrell et al. got 0.65-0.95) | Turrell et al. (2019) |

**Implementation:**
1. Pull BLS OES data for SOC 15-1252 (Software Developers) and SOC 15-1256 (Software Quality Assurance): employment by state and industry concentration.
2. Pull JOLTS information sector series from FRED.
3. Compute our distributions on the cleaned unified dataset.
4. Run correlations and report in a representativeness table.

**Do this separately for each of the three datasets.** They have different selection mechanisms and may have different representativeness profiles. The scraper metadata (search_query, query_tier, search_metro_name) enables richer diagnostics for the scraped data than was possible with Kaggle.

### 13b. Cross-dataset comparability: Are the three datasets measuring the same thing?

**Why a reviewer asks:** "You are comparing datasets with unknown/different collection methods. Any difference you find could be an artifact of the data, not the labor market."

This is the single most important validation. We must demonstrate that differences between 2024 Kaggle data and 2026 scraped data reflect real labor market changes, not measurement artifacts. With three datasets, comparability checks run pairwise:

- **arshkon vs. asaniczka** (both 2024, different months): establishes within-period baseline consistency
- **arshkon vs. scraped** (April 2024 vs. March 2026): primary temporal comparison
- **asaniczka vs. scraped** (January 2024 vs. March 2026): secondary temporal comparison (but asaniczka lacks entry-level labels and company metadata)

**Tests to run (for each pairwise comparison):**

| Test | What it measures | What we hope to see | What would be concerning |
|---|---|---|---|
| **Description length distribution** | KS test on `description_core` character counts | p > 0.05 OR explainable difference | Large difference that could be scraping artifact |
| **Company overlap** | Jaccard similarity of company names across datasets | > 0.20 for top-100 companies | Complete non-overlap suggesting different market segments |
| **Geographic distribution** | Chi-squared test on state-level shares | Similar state rankings, proportional differences | Completely different geographic profiles |
| **Seniority distribution (native labels)** | Compare LinkedIn's native seniority label distributions | Similar distributions (within 5pp per level) | Massive shifts that could indicate LinkedIn changed its labeling |
| **Industry distribution** | Chi-squared test on industry shares | Similar industry mix | One dataset dominated by an industry absent from the other |
| **Title vocabulary overlap** | Jaccard similarity of unique job titles | High overlap for common titles | New title categories in 2026 not present in 2024 (expected but should be quantified) |
| **Company size distribution** | KS test on employee counts | Similar distributions | One dataset overrepresenting small/large companies |

**Note on asaniczka limitations:** Several tests (entry-level distribution, company size, industry) cannot be run for asaniczka because it lacks those fields. Document which pairwise comparisons are feasible for which tests.

**Key confounders to investigate:**

**LinkedIn platform changes (2024 to 2026):** Did LinkedIn change its job posting display, categorization, search algorithm, or seniority labeling between 2024 and 2026? Platform changes create artificial differences that look like labor market changes. Check LinkedIn's published changelog, engineering blog, and press releases for relevant product updates.

**Indeed vs. LinkedIn composition effect:** The scraped data includes both LinkedIn and Indeed. The Kaggle data is 100% LinkedIn. Any Kaggle-vs-scraped difference could partly reflect Indeed-vs-LinkedIn differences, not temporal changes. **Mitigation:** Run all cross-period comparisons on the LinkedIn-only subset of scraped data first, then check whether including Indeed changes results.

**Scraper query design effect:** Our scraper runs specific queries in specific cities. The Kaggle datasets were collected with unknown queries and geographic scope. Different query strategies surface different jobs even on the same day. **Mitigation:** Use the new scraper metadata (search_query, query_tier, search_metro_name) to profile query-level composition. Document as a limitation. Compare title distributions to see if one dataset captures roles the other misses.

### 13c. Missing data audit

**Why a reviewer asks:** "With no entry-level labels in asaniczka and uneven metadata coverage across sources, how much of the comparison rests on thin or non-comparable fields?"

**Produce a missing data table:**

| Field | arshkon (n~2,604 SWE) | asaniczka (n~18,169 SWE) | Scraped LinkedIn | Scraped Indeed |
|---|---|---|---|---|
| Title | % | % | % | % |
| Description | % | % | % | % |
| Company name | % | % | % | % |
| Location | % | % | % | % |
| Seniority (native) | % | % (no entry-level) | % | N/A |
| Industry | % | N/A | % (100%) | % |
| Company size | % | N/A | % | % (~91%) |
| Date posted | % | % | % (~2.8%) | % (100%) |

**Missingness mechanism analysis (for entry-level labels):**

asaniczka contains only "Mid senior" and "Associate" seniority labels -- no entry-level. This means:
- Entry-level historical baseline relies entirely on arshkon (~385 entry-level SWE postings)
- Any claim about junior share decline must acknowledge this constraint
- Consider whether asaniczka "Associate" labels partially overlap with entry-level (investigate by cross-referencing titles)

**Decision:** Keep the core validation and exploration battery independent of salary fields. All primary analyses (seniority shifts, skill migration, divergence) should rely on fields present across the supported sources.

### 13d. Selection bias diagnostics

**Why a reviewer asks:** "Your data only captures jobs posted online, through specific queries, on specific platforms. How do you know your findings generalize?"

**The five selection mechanisms in our data:**

1. **Platform selection**: LinkedIn overrepresents BA+ professional jobs. For SWE roles specifically, coverage is high (>80% posted online per Carnevale et al. 2014). But control occupations (nursing, civil engineering) have lower online posting rates -- this confounds cross-occupation comparisons.

2. **Algorithm selection**: LinkedIn's ranking algorithm optimizes for engagement, not representativeness. Promoted (paid) posts get 3-5x visibility. Our scraper captures what the algorithm surfaces, not a random sample.

3. **Scraper selection**: Our query x metro x results-per-query design creates deterministic gaps:
   - Max 25 results per query-city combo means we miss long-tail postings
   - Metro selection misses smaller metros entirely
   - Query tier design means roles that don't match any query are excluded
   - The new scraper metadata (search_query, query_tier, search_metro_name) allows us to diagnose these gaps directly

4. **Employer selection**: Staffing companies (Lensa, Dice, etc.) inflate some companies' representation. DataAnnotation had 168 Kaggle SWE postings (5.4%) -- likely an outlier.

5. **Temporal selection (volatility bias)**: Daily scraping oversamples longer-lived postings (Foerderer 2023). A job open for 60 days is 60x more likely to appear in any daily scrape than a 1-day posting. This biases toward hard-to-fill roles.

**Tests to run:**

| Test | What it checks | How |
|---|---|---|
| **Covariate balance (ASMD)** | Whether scraped data distributions match BLS benchmarks | Meta's `balance` package. ASMD < 0.1 is acceptable per Stuart et al. (2013) |
| **Company size distribution vs. BLS** | Whether we over-represent large firms | Compare our company size distribution against Census SUSB for NAICS 5112 (Software Publishers) |
| **Geographic coverage map** | Whether our metro design biases results | Plot SWE postings by metro. Compare against OES metro-level SWE employment |
| **Query saturation check** | Whether 25 results/query is enough | For key queries, re-scrape with higher limits (50, 100) and compare distributions |
| **Query-tier composition** | Whether tier structure biases seniority/role mix | Use search_query and query_tier metadata to profile what each tier captures |
| **Posting duration analysis** | Whether we oversample long-lived postings | If `date_posted` is available, compute posting duration distribution. Compare against Foerderer (2023) benchmarks |

**Covariate balance protocol:**

```python
from balance import Sample
sample = Sample.from_frame(scraped_df[['seniority', 'company_size', 'industry', 'state']])
target = Sample.from_frame(bls_benchmark_df[['seniority', 'company_size', 'industry', 'state']])
adjusted = sample.set_target(target).adjust()
# Reports ASMD per covariate -- threshold: ASMD < 0.1
```

If ASMD > 0.1 for key covariates, apply inverse-probability-of-selection weighting (IPSW) to reweight our sample toward the BLS benchmark. Report results both with and without reweighting.

### 13e. Classifier validation

**Why a reviewer asks:** "Your entire study depends on correctly classifying jobs as SWE and correctly imputing seniority. How accurate are these classifiers?"

**SWE detection validation (3-tier):**

1. Sample 500 postings from each dataset: 250 classified as SWE + 250 classified as non-SWE (enriched with borderline titles containing "engineer", "developer", "software", "tech" that didn't match the pattern).
2. **Tier 2 (LLM):** LLM labels each as SWE / SWE-adjacent / non-SWE with reasoning. Prompt includes title, company, first 400 chars of description.
3. **Tier 3 (Human):** Annotator reviews LLM labels, correcting disagreements. Focus on borderline cases. Compute kappa between LLM and human.
4. Report precision, recall, F1 against the corrected gold standard.

**Seniority classifier validation (3-tier):**

1. Sample 500 postings stratified by: source (arshkon, asaniczka, scraped), predicted seniority (entry/mid/senior), and title ambiguity.
2. **Tier 2 (LLM):** Higher-quality model pre-labels seniority from title + description with reasoning, because seniority is a primary analysis variable.
3. **Tier 3 (Human):** Annotator corrects LLM labels. Compute kappa between LLM and human (target >= 0.80). Adjudicate disagreements.
4. Evaluate our imputer against the corrected gold standard. Report per-class precision/recall/F1.
5. **Critical check:** Run the same classifier on arshkon data where LinkedIn native labels exist. Compare our imputation against LinkedIn's labels. If our classifier agrees with LinkedIn at a different rate for different seniority levels, that differential error biases cross-period comparisons.
6. **asaniczka-specific check:** Since asaniczka has no entry-level labels, test whether any postings that our classifier assigns to entry-level in arshkon would be labeled "Associate" or "Mid senior" by asaniczka's schema. This identifies potential systematic mislabeling.

**Classifier temporal stability:**

Our seniority classifier was designed from 2026 posting conventions. It may perform differently on 2024 data if title conventions changed. **Test:** Compute per-class accuracy on arshkon (where native labels are available) and on scraped LinkedIn (where native labels are available). If accuracy differs significantly between periods, the classifier introduces a temporal artifact.

### 13f. Distribution comparisons for key analysis variables

**Why a reviewer asks:** "Before I believe your cross-period findings, show me the raw distributions. Are you comparing normal distributions? Skewed? Bimodal?"

For each key variable, produce distribution plots (histograms or KDEs) side by side for all three datasets, and run formal distribution comparison tests:

| Variable | Test | Why |
|---|---|---|
| Description length (chars) | KS test + QQ plot | RQ1 scope inflation proxy -- must rule out scraping artifact |
| Description length after boilerplate removal | KS test + QQ plot | The apples-to-apples version |
| Seniority distribution | Chi-squared test | RQ1 core metric |
| Company size | KS test | Composition control |
| Word count of requirements section only | KS test | More targeted scope inflation measure than full description |
| Number of distinct skills mentioned | KS test | Skill breadth index |
| Years of experience required | KS test | Direct seniority requirement measure |
| Remote work rate | Proportion test | Compositional difference |

**Interpretation framework:** A statistically significant difference is NOT automatically evidence of labor market change. It could also indicate scraping method differences, platform changes, company composition differences, or seasonal variation. For each significant difference, attempt to decompose it: how much is explained by composition (different companies, industries, geographies) vs. within-composition change?

### 13g. Compositional analysis: Is the comparison apples-to-apples?

**Why a reviewer asks:** "Maybe the seniority distribution shifted not because junior jobs disappeared, but because your 2026 scraper happened to capture a different set of companies than the 2024 Kaggle dataset."

**Decomposition approach:**

1. **Company overlap analysis:** Identify companies appearing in both arshkon and scraped datasets. For the overlapping set, compare seniority distributions. If the shift holds within the same companies, it's not a composition effect.

2. **Industry-controlled comparison:** Compare seniority distributions within matched industries (e.g., SWE postings in "Technology/Information" only, excluding healthcare SWE, finance SWE, etc.). Only feasible for arshkon (which has industry via companion files) and scraped LinkedIn (which has industry at 100%).

3. **Geography-controlled comparison:** Compare within matched metros (e.g., San Francisco SWE only, NYC SWE only).

4. **Company-size-controlled comparison:** Compare within matched size bands (e.g., large companies >10K employees only). Only feasible for arshkon (via companion files) and scraped Indeed (91% company_size).

5. **Oaxaca-Blinder decomposition** (if warranted): Formally decompose the cross-period difference in any outcome (seniority share, skill prevalence) into:
   - A composition effect (different mix of companies/industries/geographies)
   - A within-composition effect (same companies posting differently)

### 13h. Company concentration analysis and normalization

**Why a reviewer asks:** "If a handful of large employers dominate your data, your findings might reflect their hiring patterns rather than the market."

**Diagnostic steps:**

1. **Compute concentration metrics per dataset:**
   - Herfindahl-Hirschman Index (HHI): Sum of squared posting-share per company. HHI > 0.15 = moderately concentrated, > 0.25 = highly concentrated.
   - Top-5 / top-10 / top-20 company share of total SWE postings
   - Gini coefficient of company posting counts

2. **Identify dominant companies and audit them:**
   - For any company with >3% of SWE postings in either dataset, LLM reviews 20 postings per company to check for crowdwork, template-only, or aggregator patterns.
   - Flag companies that are functionally aggregators or crowdwork platforms even if not in the AGGREGATORS list.

3. **Within-company vs. between-company decomposition:**
   - For companies appearing in BOTH arshkon and scraped datasets, compare their seniority distributions across periods.
   - Compute the cross-period seniority shift (a) on the full sample, (b) on overlapping-companies-only, and (c) on the non-overlapping sample. If (b) shows the same shift as (a), company composition is not driving the finding.

4. **Company-capped analysis (sensitivity):**
   - Cap each company at N postings (e.g., N = 10 or N = median company count) to prevent any single company from dominating.
   - Re-run key analyses on the capped sample.

5. **Company-level fixed effects (for regression analyses):**
   - Include company fixed effects in any regression model to absorb between-company variation and isolate within-company changes over time.
   - Only works for companies appearing in both periods -- report the overlap rate.

6. **Exclusion sensitivity tests:**
   - Re-run analyses excluding the top-5 companies from each dataset
   - Re-run excluding all aggregators/staffing companies
   - Re-run excluding DataAnnotation specifically

**Reporting:** Include a company concentration table in the methodology section:

| Metric | arshkon SWE | asaniczka SWE | Scraped SWE |
|---|---|---|---|
| Unique companies | X | X | X |
| Top-1 company share | X% | X% | X% |
| Top-5 share | X% | X% | X% |
| Top-10 share | X% | X% | X% |
| HHI | X | X | X |
| Company overlap (Jaccard, pairwise) | | | |

### 13i. Temporal stability and seasonality checks

**Why a reviewer asks:** "With snapshots from different months (January 2024, April 2024, March 2026), how do you separate genuine structural change from normal seasonal or cyclical variation?"

**Checks:**

1. **JOLTS seasonal pattern:** Plot JOLTS information sector openings by month. Show that January-to-April variation is small relative to the cross-year change we observe. If BLS data shows a typical seasonal swing of +/-5% but we observe a 20% shift in junior share, seasonality alone cannot explain it.

2. **Within-period stability (scraped data):** Compute daily seniority distributions from scraped data and test for day-to-day stability. If the distribution is stable within a 2-week window, a 1-2 month seasonal offset is unlikely to drive findings.

3. **arshkon-asaniczka consistency:** arshkon covers April 2024; asaniczka covers January 2024. Compare their seniority and skill distributions (where both have labels). If they agree, within-2024 seasonal variation is negligible. If they disagree, seasonal effects need further investigation.

4. **External triangulation:** Compare findings against Revelio Labs hiring and openings trends, which have monthly resolution across 2021-2026. Do Revelio trends show a gradual decline or a discrete break? Does the slope accelerate around late 2025?

### 13j. Power analysis: Do we have enough data?

**Why a reviewer asks:** "With only ~385 entry-level SWE postings in arshkon (the only historical source with entry-level labels), do you have statistical power to detect a meaningful difference?"

**Key power calculations needed:**

| Analysis | Effect size to detect | Sample sizes | Estimated power |
|---|---|---|---|
| Junior share change (chi-squared) | 5pp shift (e.g., 12% to 7%) | arshkon ~2,604 vs. scraped ~N | Compute |
| Description length change (Mann-Whitney) | Cohen's d = 0.2 (small) | ~385 entry-level arshkon vs. N entry-level scraped | Compute |
| Skill prevalence change (proportion test) | 5pp shift in skill mention rate | Same | Compute |
| Cross-occupation comparison | Interaction effect | SWE ~N vs. Control ~N | Compute |

**arshkon entry-level SWE is the binding constraint:** Only ~385 arshkon SWE postings carry an entry-level label. After imputation, this might rise to 500-700, but it is still small. Compute the minimum detectable effect size given this sample.

**asaniczka cannot help with entry-level power** because it lacks entry-level labels entirely. Document this limitation.

### 13k. Robustness pre-registration: What specifications will we test?

**Why a reviewer asks:** "You could have tried 50 different specifications and reported the one that worked. How do I know you didn't?"

Define the specification space BEFORE looking at results:

**SWE definition variants:**
1. Narrow: Current `SWE_PATTERN` (core SWE titles only)
2. Broad: Add "data scientist", "data analyst", "product engineer"
3. Excluding adjacent: Remove "data engineer", "ML engineer" (these may have different dynamics)

**Seniority classification variants:**
1. LLM-classified (primary, where available), else rule-based imputed
2. LinkedIn native labels where available, imputed where missing
3. Description-only classifier (ignore titles)

**Dedup variants:**
1. Strict: Exact match on (title, company, location)
2. Standard: Near-dedup with similarity >= 0.70
3. Loose: Near-dedup with similarity >= 0.50

**Sample variants:**
1. Full sample
2. LinkedIn only (excludes Indeed composition effect)
3. LinkedIn + Indeed pooled
4. Excluding aggregator-like employers
5. Metro-balanced subsamples
6. Excluding top-5 most common companies (reduces concentration bias)

**Observation-level variants:**
1. Canonical postings (deduplicated)
2. Daily observations (from `unified_observations.parquet`)

**Key findings must hold across all defensible specifications.** Use the `specification_curve` package to visualize this. If a finding is fragile (holds under some specifications but not others), it is reported as suggestive, not conclusive.

### 13l. Placebo and falsification tests (pre-registration)

**Note:** This section defines the placebo tests. Stage 14 exploration produces results; placebos are executed after main results are available.

**Why a reviewer asks:** "Maybe your method finds 'structural change' in any two snapshots, regardless of whether anything actually changed."

**Placebos to pre-register:**

1. **Control occupation placebo:** Run the same seniority-shift analysis on control occupations (civil engineering, nursing, mechanical engineering). If they show the same "structural change" as SWE, the finding is confounded by macro trends or measurement artifacts, not AI-specific restructuring.

2. **Within-arshkon time-split placebo:** Split arshkon into early April vs. late April. Run the same analysis across the two halves. If we find a "shift" within a 2-week window, our method is detecting noise.

3. **Shuffled-label test:** Randomly permute the dataset labels (arshkon vs. scraped) and re-run the analysis 10,000 times. The observed effect size should exceed 95% of permuted effect sizes.

4. **Within-scraped week-over-week:** Split scraped data into week 1 vs. week 2. Run the same analyses. Expect null results.

5. **Null-effect occupations:** Identify occupations with no theoretical reason to be affected by AI coding agents (e.g., registered nurses, civil engineers). Run the full analysis pipeline on these as negative controls. Expect null results.

### 13m. Bias threat summary table

Produce a consolidated table for the methodology section:

| Bias | Direction | Magnitude estimate | Mitigation | Residual risk |
|---|---|---|---|---|
| Platform selection (LinkedIn overrepresents tech/professional) | Favors SWE coverage; underrepresents control occupations | ~11pp for tech occupations (Hershbein & Kahn) | Post-stratification against OES | Low for SWE; moderate for controls |
| Algorithm selection (promoted posts) | Unknown direction | Unknown | Cannot correct; acknowledge | Moderate |
| Scraper query design (results x cities) | Misses long-tail postings | Unknown; diagnosable via search metadata | Query saturation check; query-tier profiling | Moderate |
| Aggregator contamination | Inflates some company counts; adds boilerplate | 9% of scraped, ~15% of Kaggle SWE | Flag and sensitivity analysis | Low after flagging |
| Temporal selection (volatility bias) | Oversamples long-lived postings | 60:1 for 60-day vs. 1-day postings (Foerderer) | Report duration distribution; consider IPW | Moderate |
| Kaggle provenance unknown | Could bias anything | Unknown | Treat as stated limitation | High (irreducible) |
| Ghost jobs | Inflates entry-level tech postings | 18-27% of all postings (CRS 2025) | Flag and sensitivity analysis | Moderate |
| Platform changes (2024 to 2026) | Could create artificial differences | Unknown | Investigate LinkedIn changelog; run LinkedIn-only comparison | Moderate (irreducible) |
| Company composition shift | Could drive apparent seniority shift | Unknown until tested | Oaxaca-Blinder decomposition; within-company comparison | Low after decomposition |
| Seasonal offset (different months across datasets) | Could inflate/deflate metrics | Typically small for adjacent months | JOLTS seasonal comparison; arshkon-asaniczka cross-check | Low |
| asaniczka missing entry-level labels | Eliminates one baseline source for junior analysis | All entry-level baseline comes from arshkon (~385 postings) | Acknowledge; compute power given constraint | Moderate |

---

## Spot-check protocol

Manual review is non-negotiable for publication quality. All spot-checks use the 3-tier review protocol (rules -> LLM -> human). See `plan-preprocessing.md` for the full spot-check table with tier assignments and sample sizes.

---

## Stage 14: Exploration and discovery

This phase builds intuition about the data before formal hypothesis testing. Every exploration here is designed to either (a) generate visualizations and tables that go directly into the paper, (b) surface unexpected patterns that refine our research questions, (c) validate that the preprocessed data behaves as expected before committing to expensive analyses, or (d) produce artifacts for interview elicitation (RQ4).

**Key tool choices for this stage:**

| Tool | What it does | Why we use it here |
|---|---|---|
| [**JobBERT-v2**](https://huggingface.co/TechWolf/JobBERT-v2) | Sentence transformer fine-tuned on 5.5M job title-skill pairs (MPNet base, 1024d). **Max 64 tokens -- titles only.** | Use for **title-level** tasks: classification, title clustering, title similarity. NOT for full descriptions (64-token limit). |
| **Description model** (selected via Stage 5d benchmark) | General-purpose sentence transformer (`all-mpnet-base-v2` or `e5-large-v2`). 384-512 token context. | Use for **description-level** tasks: topic modeling (BERTopic), content convergence, drift measurement, embedding space exploration. |
| [**BERTopic**](https://bertopic.com/) | Neural topic modeling (embedding -> UMAP -> HDBSCAN -> c-TF-IDF). Supports dynamic topic modeling over time. | Discovers emergent skill clusters we did not think to look for. For exploration and robustness, not headline claims. |
| [**BERTrend**](https://github.com/rte-france/BERTrend) | Runs BERTopic per time slice, merges across windows, classifies topics as noise / weak signal / strong signal. | Detects genuinely new topics that emerge between 2024 and 2026. |
| [**Fightin' Words**](https://github.com/Wigder/fightin_words) | Log-odds-ratio with Dirichlet prior for pairwise corpus comparison. Produces both effect size and z-score per word. | High-value early text-comparison tool (per 6-methods-learning.md). Statistically rigorous, handles corpus size imbalance. |
| [**Scattertext**](https://github.com/JasonKessler/scattertext) | Interactive HTML visualization of distinguishing terms between two corpora. | Produces publication-quality figures and lets us visually inspect what drives corpus differences. |
| [**KeyBERT**](https://github.com/MaartenGr/KeyBERT) | Keyword/keyphrase extraction using BERT embeddings + cosine similarity. | Extracts the most representative terms from each posting or group of postings without a predefined dictionary. |
| [**Nesta OJD Skills Library**](https://github.com/nestauk/ojd_daps_skills) | End-to-end pipeline: extract skill phrases from job ads, map to ESCO or Lightcast taxonomy. | Built specifically for job ad analysis. Handles the full pipeline from raw text to structured skill tags. |

**Methods guidance (from 6-methods-learning.md):**
- Fightin' Words is a high-value early text-comparison tool. Run it first.
- Topic models (BERTopic/STM) are for exploration and robustness only, not headline claims.
- LLM-assisted annotation should only happen after human codebook design and validation.
- Reflexive thematic analysis is the right approach for interviews (RQ4).

### 14a. Raw data inspection (3-tier: rules -> LLM -> human)

**Before any automated analysis, look at the data.** Automated methods can produce plausible-looking results from garbage data.

**Tier 1 -- Rule-based screening (full dataset):**
- Flag descriptions < 100 chars, > 15,000 chars, or with non-ASCII majority
- Flag postings where seniority label contradicts years-of-experience in description
- Flag entry-level titles with 5+ years required
- Flag company name mismatches (aggregator name vs. description employer)
- Output: CSV of flagged rows with flag reasons

**Tier 2 -- LLM bulk review (500 postings):**
Sample 500 postings (stratified: 100 arshkon SWE, 100 asaniczka SWE, 150 scraped SWE, 50 non-SWE, 100 extreme values from Tier 1 flags). For each, LLM assesses: is_real, seniority_match, desc_quality, requirements_realistic, concerns, summary.

- Output: JSONL file with LLM assessments for all 500 postings
- Aggregate stats: % flagged per issue type, by dataset

**Tier 3 -- Human review (targeted, ~50 postings):**
- Review all postings LLM flagged as problematic (spam, mismatch, ghost-job)
- Review 10 random "clean" postings to validate LLM accuracy
- **Side-by-side comparison (10 matched pairs):** Find 10 companies in both arshkon and scraped datasets. Pull one SWE posting from each period. Compare descriptions: structure changes, requirement changes, AI mentions. Quote specific examples in the paper.

**Output:** `data/quality_review.md` -- annotated examples tagged with `[boilerplate]`, `[ghost-job?]`, `[skill-inflation]`, `[good-example]`, `[misclassified]`. Plus `data/llm_review_results.jsonl` for the full LLM assessment dataset.

### 14b. Embedding space exploration

**Goal:** Visualize how job postings cluster in semantic space and whether clusters align with our seniority/occupation categories.

**Implementation:**

1. **Embed all SWE postings** using the description-level model selected in Stage 5d (likely `all-mpnet-base-v2` or `e5-large-v2`). Use `title + first 400 words of description_core` as input. For title-only tasks, use JobBERT-v2 separately.

2. **UMAP projection to 2D:**
   ```python
   import umap
   reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine')
   coords = reducer.fit_transform(embeddings)
   ```

3. **Visualization layers** (produce multiple views of the same embedding space):
   - Color by **seniority** (entry/mid/senior) -- do seniority levels form distinct clusters, or do they overlap?
   - Color by **period** (2024 vs. 2026) -- do the two periods occupy different regions, or are they interleaved?
   - Color by **source** (arshkon vs. asaniczka vs. scraped) -- do collection methods create artificial clustering?
   - Color by **company** (top-10 companies highlighted) -- do individual companies form tight clusters (standardized templates)?
   - Color by **topic** (from BERTopic, step 14d) -- visual confirmation that topic clusters are semantically coherent.

4. **Quantitative embedding analysis:**
   - **Junior-senior centroid distance:** Compute the cosine distance between the average junior embedding and the average senior embedding, separately for each period. If this distance shrank between 2024 and 2026, junior roles are converging toward senior roles in content (RQ1 evidence). **This produces the junior-senior embedding similarity figure.**
   - **Within-seniority variance:** Compute the average pairwise cosine distance within each seniority level. If junior postings have higher variance in 2026 than 2024, the category is becoming more heterogeneous.
   - **Cross-period drift per seniority:** For each seniority level, compute the cosine distance between the 2024 centroid and the 2026 centroid. If junior roles drifted more than senior roles, that is directional evidence for RQ1.

### 14c. Corpus comparison -- Fightin' Words and Scattertext

**Goal:** Identify the specific words and phrases that statistically distinguish different groups (junior vs. senior, 2024 vs. 2026, SWE vs. control). This is the highest-value early text-comparison step.

**Comparisons to run (6 total):**

| Comparison | What it tests | RQ |
|---|---|---|
| Junior 2024 vs. Junior 2026 | How junior role language changed | RQ1 |
| Senior 2024 vs. Senior 2026 | How senior role language changed | RQ1 (senior redefinition) |
| Junior 2024 vs. Senior 2024 | Baseline junior-senior gap | RQ1, RQ2 |
| Junior 2026 vs. Senior 2026 | Current junior-senior gap | RQ1, RQ2 |
| Junior 2026 vs. Senior 2024 | Are 2026 juniors linguistically similar to 2024 seniors? | RQ1 (redefinition hypothesis) |
| SWE 2026 vs. Control 2026 | SWE-specific vs. general labor market language | Supporting evidence |

**Implementation:**

1. **Fightin' Words** (quantitative):
   ```python
   from fightin_words import FWExtractor
   fw = FWExtractor(ngram_range=(1, 3), min_df=5)
   results = fw.fit_transform(corpus_a, corpus_b)
   # Returns (word, log-odds-ratio, z-score) for every n-gram
   # Sort by |z-score| to find the most distinguishing terms
   ```
   Use `description_core` (boilerplate-stripped). Run on unigrams, bigrams, and trigrams separately. Filter to |z-score| > 3.0 for statistical significance after Bonferroni correction.

2. **Scattertext** (visual):
   Produces an interactive HTML plot. Each dot is a word. Position = relative frequency in each corpus. This is both an analysis tool and a publication figure.

3. **What to look for:**
   - **AI/LLM terms emerging:** Do "LLM", "AI agent", "prompt engineering", "RAG", "vector database" appear in 2026 but not 2024?
   - **Management terms declining in senior roles:** Do "mentorship", "coaching", "team leadership", "hiring" shift downward in 2026 senior postings? (RQ1 senior redefinition)
   - **Senior terms appearing in junior roles:** Do "system design", "architecture", "end-to-end", "cross-functional" appear more in 2026 junior postings than 2024? (RQ2 task migration)
   - **Years-of-experience inflation:** Do "3+ years", "5+ years" appear more frequently in 2026 entry-level postings? (ghost job / scope inflation signal)

### 14d. Topic discovery -- BERTopic and BERTrend

**Goal:** Discover latent topic structure and track which topics are emerging, stable, or declining between periods. This catches skill categories and role archetypes that our predefined keyword lists miss. Per 6-methods-learning.md, topic models are for exploration and robustness, not headline claims.

**Implementation:**

1. **Base BERTopic on full SWE corpus (both periods combined):**
   ```python
   from bertopic import BERTopic
   from sentence_transformers import SentenceTransformer

   embedding_model = SentenceTransformer("all-mpnet-base-v2")  # description-level model
   topic_model = BERTopic(
       embedding_model=embedding_model,
       min_topic_size=30,
       nr_topics="auto",
       verbose=True
   )
   topics, probs = topic_model.fit_transform(descriptions)
   ```

   **Critical:** Use `description_core` (boilerplate-stripped). EEO statements and benefits sections are the #1 source of spurious topics.

2. **Dynamic topic modeling (topics over time):**
   With only 2-3 time points (January 2024, April 2024, March 2026), resolution is limited -- but the signal we are looking for (new AI topics, declining management topics) should be detectable.

3. **BERTrend for emerging signal detection:**
   BERTrend classifies each topic as **noise**, **weak signal**, or **strong signal** based on popularity trends. Designed for exactly our use case: detecting whether AI-orchestration skills are a weak signal in 2024 that became a strong signal by 2026.

4. **Topic-level analysis for each RQ:**

   | RQ | What to look for in topics | Expected signal |
   |---|---|---|
   | RQ1 | Topics that appear in senior postings in 2024 but migrate to junior postings in 2026 | Topics like "system design", "architecture decisions" |
   | RQ2 | The temporal order in which skill-topics appear in junior postings | Sequence: cloud -> CI/CD -> system design -> AI tools |
   | RQ1 (senior) | Topics in senior postings: management-heavy in 2024, AI-heavy in 2026 | "Team leadership, hiring" declining; "AI integration, agent orchestration" emerging |
   | Supporting | Whether the same topic shifts appear in control occupations | They should NOT (if they do, it's macro confounding) |

5. **Guided BERTopic** (semi-supervised):
   Use RQ-derived skill categories as seed topics:
   ```python
   seed_topic_list = [
       ["system design", "architecture", "distributed systems", "scalability"],
       ["AI", "LLM", "prompt engineering", "RAG", "agent", "copilot"],
       ["mentorship", "coaching", "team lead", "hiring", "performance review"],
       ["CI/CD", "deployment", "infrastructure", "DevOps", "kubernetes"],
       ["testing", "QA", "quality assurance", "test automation"],
   ]
   topic_model = BERTopic(seed_topic_list=seed_topic_list, ...)
   ```

### 14e. Structured skill extraction

**Goal:** Extract structured, taxonomy-mapped skills from free-text descriptions. This goes far beyond keyword matching.

**Why this matters:** Our current skill analysis uses a hand-coded list of 16 keywords. This misses skills we did not think of, conflates different uses of the same word, and cannot handle synonyms ("k8s" = "Kubernetes" = "container orchestration").

**Implementation -- two complementary approaches:**

1. **Nesta OJD Skills Library** (taxonomy-mapped):
   Maps to ESCO (13,890 skills) or Lightcast Open Skills taxonomy. Gives us a controlled vocabulary for cross-period comparison.

2. **KeyBERT** (unsupervised, catches what taxonomies miss):
   Using JobBERT-v2 as the backbone means keyphrases are scored by relevance to the job domain. Catches emerging terms (e.g., "agentic workflow", "vibe coding") that are not in any taxonomy yet.

3. **Skill prevalence analysis (replaces hand-coded keyword lists):**
   - For each ESCO-mapped skill, compute prevalence by seniority x period
   - Identify skills with the largest prevalence change between periods
   - Identify skills that migrated from senior-only to junior+senior
   - Produce the **requirement migration heatmap** (paper figure 5) grounded in a standard taxonomy rather than researcher-selected keywords

### 14f. Seniority boundary analysis

**Goal:** Understand where the junior/senior boundary actually lies in the data, and whether it moved between periods. This addresses the core of RQ1.

**Implementation:**

1. **Embedding-based seniority boundary:**
   - Train a simple logistic regression on description embeddings to predict seniority (junior vs. senior) using 2024 data only
   - Apply this classifier to 2026 data. If 2026 "junior" postings are classified as "senior" by the 2024 model at a higher rate than 2024 "junior" postings, the boundary has shifted.
   - Report: % of 2026 junior postings that the 2024-trained model classifies as senior (the "redefinition rate")

2. **Decision boundary visualization:**
   - In the UMAP space from 14b, draw the decision boundary of the 2024-trained seniority classifier
   - Overlay 2026 junior postings. How many fall on the "senior" side of the 2024 boundary?
   - This is a strong visual for the paper.

3. **Feature importance for the boundary:**
   - What words/skills most strongly predict "senior" vs. "junior" in the 2024 model?
   - Which of those features are now present in 2026 junior postings?
   - Use SHAP values or logistic regression coefficients for interpretability.

### 14g. Requirements section parsing

**Goal:** Extract structured data from the requirements/qualifications section of job descriptions.

**Why this is separate from full-text analysis:** The requirements section is the most decision-relevant part of a job posting for applicants and for our research. "Scope inflation" (RQ1) should be measured by what is required, not by the "About Us" section.

**Implementation:**

1. **Section extraction** (from Stage 3 boilerplate removal):
   - Parse `description_core` into sections: responsibilities, requirements/qualifications, nice-to-haves
   - Analyze requirements sections separately from full descriptions

2. **Structured requirement extraction:**
   - **Years of experience:** Extract all "X+ years" patterns. Compute min, max, and median years required per seniority level x period. Test for inflation.
   - **Education requirements:** Extract degree mentions (BS, MS, PhD). Compute degree distribution per seniority x period.
   - **Technology requirements:** Count distinct technologies mentioned in requirements. Are junior roles in 2026 requiring more technologies than in 2024?
   - **Soft skill requirements:** Extract management/leadership/communication mentions from requirements. Are junior roles now requiring "cross-functional collaboration" and "stakeholder management"?

3. **Requirement count as scope metric:**
   - Count the number of bullet points or distinct requirements per posting
   - Compare this count across seniority x period
   - This is a more targeted "scope inflation" metric than description word count

### 14h. Temporal drift measurement

**Goal:** Quantify how much the posting content changed between 2024 and 2026, and characterize the direction of change.

**Implementation:**

1. **Corpus-level embedding drift:**
   Compute centroid embeddings for each period. Report cosine distance overall and per seniority level.

2. **Vocabulary drift (JSD):**
   - Build unigram frequency distributions for each period
   - Compute Jensen-Shannon divergence between them
   - Do this for the full SWE corpus, and separately for junior-only and senior-only
   - JSD for junior postings vs. JSD for senior postings: if junior postings changed more than senior, that is RQ1 evidence

3. **Nearest-neighbor stability:**
   - For each posting in the overlap set (companies appearing in both periods), find its k=10 nearest neighbors
   - What fraction of neighbors are from the same period vs. the other period?
   - If 2026 junior postings' nearest neighbors are mostly 2024 senior postings, that is strong evidence of content convergence

4. **Keyword emergence/disappearance:**
   - Terms appearing in 2026 at >1% prevalence but absent from 2024 (or <0.1%): emerging requirements
   - Terms appearing in 2024 at >1% prevalence but absent from 2026: declining requirements
   - Do this separately for junior and senior postings

### 14i. Company-level patterns

**Goal:** Understand whether observed shifts are driven by within-company changes (same companies posting differently) or between-company changes (different companies dominating).

**Implementation:**

1. **Company overlap set analysis:**
   - Identify companies appearing in both arshkon and scraped datasets
   - For overlapping companies: compare their seniority distributions, skill mentions, description length across periods
   - This is the within-company change signal -- not confounded by composition

2. **Company archetypes (via clustering):**
   - Cluster companies by their posting profiles (average embedding, seniority mix, skill distribution)
   - Are there distinct "types" of SWE employers? (e.g., FAANG-style, startup-style, consulting-style, government)
   - Did the relative share of these archetypes change between periods?

3. **Firm-size effects:**
   - Split by company size bands (1-50, 50-500, 500-5000, 5000+)
   - Run all key metrics within each size band
   - Does scope inflation show equally in startups and large enterprises, or is it concentrated?

### 14j. Ghost job and anomaly profiling

**Goal:** Characterize the ghost job phenomenon and understand whether it biases our findings.

**Implementation:**

1. **Ghost job feature analysis:**
   - For postings flagged as ghost-risk in Stage 8, profile them: which companies, which seniority levels, which geographies, how their descriptions differ from non-ghost postings
   - Do ghost jobs have systematically different skill requirements (more inflated)?

2. **Anomaly detection:**
   - Use isolation forest or DBSCAN on the embedding space to identify outlier postings
   - Manually review the outliers: spam, duplicate templates, non-English, non-US, misclassified occupation
   - Report the anomaly rate and exclude from analysis with sensitivity check

### 14k. Cross-occupation comparison

**Goal:** Establish that observed changes are SWE-specific, not part of a broader labor market trend (supporting evidence for RQ1-RQ2).

**Implementation:**

1. **Run the same exploration on control occupations:**
   - Embedding space, Fightin' Words, topic modeling, skill extraction -- all on civil engineering, nursing, mechanical engineering postings
   - Default production outputs keep controls on rule-based `description_core`; run a separate control-extraction sensitivity job if cross-occupation text analyses need `description_core_llm`
   - The key question: do control occupations show the same patterns (AI skill emergence, scope inflation, seniority compression)?
   - If they do, SWE findings are confounded. If they don't, we have comparative evidence.

2. **SWE-adjacent occupation analysis:**
   - Data scientist, product manager, UX designer -- these are AI-exposed but not SWE
   - Do they show similar restructuring patterns? Tests whether the effect is specific to coding or broader to tech.

3. **Cross-occupation embedding distance:**
   - How far apart are SWE, SWE-adjacent, and control postings in embedding space?
   - Is the distance between SWE and SWE-adjacent shrinking (role convergence)?

### 14l. Employer-requirement / worker-usage divergence (RQ3)

**Goal:** Compare posting-side AI requirements against worker-side observed AI usage benchmarks to test the anticipatory restructuring hypothesis.

**Implementation:**

1. **Posting-side AI requirement rate:**
   - Compute the share of SWE postings mentioning AI/LLM/agent/copilot requirements, by period
   - Break out by seniority level

2. **Worker-side AI usage benchmark:**
   - Use Anthropic occupation-level AI usage data as the primary external benchmark
   - Map to SOC codes comparable to our SWE definition

3. **Divergence index:**
   - Compute the gap between employer-side requirement rate and worker-side observed usage rate
   - If employer requirements outpace observed workplace usage, this is consistent with anticipatory restructuring
   - Frame carefully: the two measures are not directly interchangeable (different units, populations, time coverage)

4. **Output:** The **posting-usage divergence chart** (paper figure 6) and the **posting-usage divergence interview artifact** for RQ4 elicitation.

---

## Interview elicitation artifacts (for RQ4)

The exploration phase must produce specific artifacts for data-prompted elicitation in interviews (from `2-interview-design-mechanisms.md`). These are not optional appendix items; they are required inputs to the qualitative study.

| Artifact | Source analysis | What to prepare | Target cohorts |
|---|---|---|---|
| **Inflated junior JD examples** | 14a (raw inspection), 14g (requirements parsing) | 3-5 real entry-level postings with system design, CI/CD, AI-tool, ownership language that exceeds expected junior scope | All cohorts |
| **Paired JDs over time** | 14i (company-level patterns) | Same company, similar role, one from arshkon 2024 and one from scraped 2026. Highlight what changed. | Seniors, hiring-side |
| **Junior-share trend plot** | 13f (distribution comparisons), Stage 14 descriptive counts | Junior SWE posting share over time, annotated with AI model release dates | Seniors, hiring-side |
| **Senior archetype chart** | 14c (Fightin' Words), 14d (BERTopic) | Management vs. orchestration language prevalence in senior postings over time | Seniors, hiring-side |
| **Posting-usage divergence chart** | 14l (RQ3 analysis) | Posting AI mention rate vs. observed AI usage benchmark (Anthropic data) | All cohorts |

**Protocol rule (from 2-interview-design-mechanisms.md):**
- Ask the open question first
- Show the artifact second
- Ask what feels real, false, missing, or overstated

---

## Sensitivity analyses

All main findings must be tested under these sensitivity specifications (from `1-research-design.md`):

1. **LinkedIn-only estimates** -- excludes Indeed composition effect; directly comparable to Kaggle
2. **LinkedIn + Indeed pooled estimates** -- tests whether adding Indeed changes conclusions
3. **Exclusion of aggregator-like employers** -- removes Lensa, Dice, DataAnnotation, etc.
4. **Metro-balanced subsamples** -- reweights to match OES metro-level SWE employment
5. **Dedupe and repost sensitivity** -- strict vs. standard vs. loose dedup thresholds
6. **Canonical postings vs. daily observations** -- tests whether duration-weighted observations change conclusions (uses `unified_observations.parquet`)
7. **Company-capped** -- caps each company at N postings to reduce concentration effects

---

## Outputs

### Paper figures (from 1-research-design.md)

| # | Figure | Primary source |
|---|---|---|
| 1 | Junior posting share and volume over time | 13f, 14h |
| 2 | Junior scope-inflation index over time | 14f, 14g |
| 3 | Senior archetype shift index over time | 14c, 14d |
| 4 | Junior-senior embedding similarity over time | 14b |
| 5 | Requirement migration heatmap by seniority and period | 14e |
| 6 | Employer-requirement / worker-usage divergence plot | 14l |
| 7 | Source-specific robustness plots | Sensitivity analyses |
| 8 | Annotated break-analysis plot with candidate release windows | Supporting analysis |

### Paper tables (from 1-research-design.md)

| # | Table | Primary source |
|---|---|---|
| 1 | Summary statistics by source, period, and seniority | 13c, 13f |
| 2 | Validation results for text measures | 13a, 13b, 13e |
| 3 | Regression estimates for junior scope inflation | Analysis plan |
| 4 | Regression estimates for senior archetype shift | Analysis plan |
| 5 | Sensitivity and robustness checks | Sensitivity analyses |
| 6 | Interview sample and mechanism summary | RQ4 interviews |

### Dataset outputs

- `unified.parquet`: canonical postings corpus
- `unified_observations.parquet`: daily observation panel
- Measurement appendix documenting dedupe, cleaning, and construct definitions

### Exploration artifacts

| Output | Type | Used in |
|---|---|---|
| Annotated raw sample | Markdown file | Qualitative examples for paper |
| UMAP embedding plots (5 color schemes) | PNG + interactive HTML | Paper figures |
| Scattertext comparisons (6 pairs) | Interactive HTML | Paper figures, appendix |
| Fightin' Words tables (6 comparisons) | CSV + sorted tables | Paper tables |
| BERTopic model + topic list | Saved model + CSV | RQ2 analysis input |
| BERTrend signal report | CSV (topic x signal strength) | RQ2 analysis input |
| ESCO-mapped skill prevalence table | Parquet | RQ2 requirement migration heatmap |
| KeyBERT emerging terms list | CSV | Appendix |
| Seniority boundary classifier | Saved model | RQ1 redefinition rate |
| Requirements section structured data | Parquet (years, degree, tech count) | RQ1, RQ2 |
| Company-level metric table | Parquet | Compositional analysis |
| Ghost job profile | Markdown + CSV | Methodology section |
| Posting-usage divergence data | CSV | RQ3 figure, interview artifact |
| Interview elicitation artifacts (5 items) | PDF/PNG | RQ4 interviews |

---

## How validation feeds into the methodology section

| Output | Section | Content |
|---|---|---|
| **Table: Data sources and coverage** | S3.1 Data | Date ranges, sample sizes, platform, selection mechanism, field availability per source |
| **Table: Missing data rates** | S3.1 Data | Per-field missingness by dataset (from 13c) |
| **Table: Representativeness** | S3.2 Validation | Our distribution vs. OES, with correlations and dissimilarity indices (from 13a) |
| **Table: Classifier performance** | S3.3 Classification | Per-class precision/recall/F1 for SWE detection and seniority imputation (from 13e) |
| **Table: Cross-dataset comparability** | S3.2 Validation | Distribution comparison tests for key variables (from 13b, 13f) |
| **Table: Bias threat summary** | S3.4 Limitations | The bias table from 13m |
| **Figure: Specification curve** | S4 Results | Effect stability across all defensible specifications (from 13k) |
| **Table: Placebo tests** | S4 Results | Null results on control occupations and random splits (from 13l) |
| **Table: Data funnel** | S3.1 Data | Raw -> cleaned -> final counts per source (from preprocessing pipeline) |
| **Table: Company concentration** | S3.2 Validation | HHI, top-N shares, overlap rates (from 13h) |

---

## Implementation order

### Phase 2: Validation (after Stages 1-12 produce `unified.parquet`)

```
Stage 13 substeps, roughly parallelizable:

13a. Representativeness checks         <- pull BLS/JOLTS benchmarks
13b. Cross-dataset comparability       <- three-way pairwise comparisons
13c. Missing data audit                <- missingno diagnostics
13d. Selection bias diagnostics        <- balance package, query-tier profiling
13e. Classifier validation             <- gold-standard annotation
13f. Distribution comparisons          <- KS tests, histograms
13g. Compositional analysis            <- company overlap, Oaxaca-Blinder
13h. Company concentration             <- HHI, capping, within-company analysis
13i. Temporal stability                <- daily variance, JOLTS seasonality, arshkon-asaniczka cross-check
13j. Power analysis                    <- sample size calculations (arshkon entry-level is binding)
13k. Robustness specification space    <- define BEFORE looking at results
13l. Placebo/falsification tests       <- control occupation null tests (execute after main results)
13m. Bias threat summary               <- consolidate all threats into one table
```

Most checks (13a-13i) can run in parallel. 13k (specification space) should be defined before 13l (placebo tests).

### Phase 3: Exploration and discovery

```
Stage 14 substeps:

14a. Raw data inspection & manual review
14b. Embedding space exploration
14c. Corpus comparison -- Fightin' Words (run early, high value)
14d. Topic discovery -- BERTopic + BERTrend
14e. Structured skill extraction
14f. Seniority boundary analysis
14g. Requirements section parsing
14h. Temporal drift measurement
14i. Company-level patterns
14j. Ghost job and anomaly profiling
14k. Cross-occupation comparison
14l. Employer-requirement / worker-usage divergence (RQ3)
```

**Priority order:** 14a -> 14c -> 14b -> 14e -> 14f -> 14g -> 14l -> 14d -> 14h -> 14i -> 14j -> 14k

Fightin' Words (14c) runs early because it is the highest-value text-comparison tool and informs all downstream analyses. RQ3 divergence analysis (14l) must complete before interviews begin, because it produces an elicitation artifact.

**After exploration, produce interview artifacts** from 14a (inflated JDs), 14c/14d (senior archetype chart), 14f/13f (junior-share trend), 14i (paired JDs), and 14l (divergence chart). These must be ready before RQ4 interview fieldwork begins.

**Iteration:** Validation or exploration results may trigger re-runs of preprocessing (Stages 1-12 in `plan-preprocessing.md`).
