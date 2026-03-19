# Exploration & Validation Plan

Date: 2026-03-19
Status: Draft — ready for review before implementation

This document covers Stage 9 (data validation) and Stage 10 (exploratory analysis). It runs on the cleaned `unified.parquet` produced by the preprocessing pipeline (`plan-preprocessing.md`). For hypothesis testing, see `plan-analysis.md`.

---

## Stage 9: Data validation

This is the core validation battery. Every check here answers a question a reviewer would ask. Results feed into the methodology section and determine whether downstream analyses are defensible.

The validation stage runs AFTER preprocessing (Stages 1-8) is complete, on the cleaned unified dataset. Some checks may trigger iteration on earlier stages (e.g., if representativeness is poor, revisit dedup thresholds or geographic filtering).

### 9a. Representativeness: Do our scraped data look like the real labor market?

**Why a reviewer asks:** "Your data is whatever LinkedIn's algorithm gave you. How do you know it reflects actual job openings?"

**Benchmark sources:**
- **JOLTS** (via FRED): Total job openings, Professional & Business Services, Information sector. Free CSV download. Establishes macro-level plausibility.
- **BLS OES** (Occupational Employment & Wage Statistics): Occupation × metro employment counts and wages. The gold standard for occupation-level validation.
- **Revelio Labs** (already ingested): SOC-15 aggregate hiring, openings, salary trends.

**Tests to run:**

| Test | What it measures | Acceptable threshold | Reference |
|---|---|---|---|
| **Occupation share correlation** | Pearson r between our occupation distribution and OES employment shares | r > 0.80 (Hershbein & Kahn got 0.84-0.98) | Hershbein & Kahn (2018) |
| **Industry share comparison** | Our SWE industry distribution vs. OES SWE industry distribution | Within 5pp per industry | OECD (2024) |
| **Geographic share correlation** | Our state-level SWE counts vs. OES state-level SWE employment | r > 0.80 | Hershbein & Kahn (2018) |
| **Dissimilarity index** | Duncan index between our occupation distribution and OES | Report and discuss (no hard cutoff) | OECD (2024) |
| **Posting volume vs. JOLTS** | Time-series correlation between our daily posting counts and JOLTS information sector openings | r > 0.60 (Turrell et al. got 0.65-0.95) | Turrell et al. (2019) |
| **Wage distribution check** | KS test comparing our salary distribution against OES wage percentiles for SOC 15-1252 (Software Developers) | p > 0.05 (non-rejection = compatible) | Hazell & Taska (2023) |

**Implementation:**
1. Pull BLS OES data for SOC 15-1252 (Software Developers) and SOC 15-1256 (Software Quality Assurance): employment by state, median wage, industry concentration.
2. Pull JOLTS information sector series from FRED.
3. Compute our distributions on the cleaned unified dataset.
4. Run correlations and report in a representativeness table.

**Do this separately for Kaggle and scraped datasets.** They have different selection mechanisms and may have different representativeness profiles.

### 9b. Cross-dataset comparability: Are Kaggle and scraped data measuring the same thing?

**Why a reviewer asks:** "You're comparing two datasets with unknown/different collection methods. Any difference you find could be an artifact of the data, not the labor market."

This is the single most important validation for our study. We must demonstrate that differences between Kaggle (April 2024) and scraped (March 2026) data reflect real labor market changes, not measurement artifacts.

**Tests to run:**

| Test | What it measures | What we hope to see | What would be concerning |
|---|---|---|---|
| **Description length distribution** | KS test on `description_core` character counts | p > 0.05 OR explainable difference | Large difference that could be scraping artifact |
| **Company overlap** | Jaccard similarity of company names across datasets | > 0.20 for top-100 companies | Complete non-overlap would suggest different market segments |
| **Geographic distribution** | Chi-squared test on state-level shares | Similar state rankings, proportional differences | Completely different geographic profiles |
| **Seniority distribution (native labels)** | Compare LinkedIn's native seniority label distributions | Similar distributions (entry/mid/senior proportions within 5pp) | Massive shifts that could indicate LinkedIn changed its labeling |
| **Industry distribution** | Chi-squared test on industry shares (after joining Kaggle companion data) | Similar industry mix | One dataset dominated by an industry absent from the other |
| **Title vocabulary overlap** | Jaccard similarity of unique job titles | High overlap for common titles | New title categories in 2026 not present in 2024 (expected but should be quantified) |
| **Company size distribution** | KS test on employee counts | Similar distributions | One dataset overrepresenting small/large companies |

**Key confounders to investigate:**

**LinkedIn platform changes (2024 → 2026):** Did LinkedIn change its job posting display, categorization, search algorithm, or seniority labeling between April 2024 and March 2026? Platform changes create artificial differences that look like labor market changes. Check LinkedIn's published changelog, engineering blog, and press releases for relevant product updates.

**Indeed vs. LinkedIn composition effect:** The scraped data is 60% LinkedIn + 40% Indeed. The Kaggle data is 100% LinkedIn. Any Kaggle-vs-scraped difference could partly reflect Indeed-vs-LinkedIn differences, not temporal changes. **Mitigation:** Run all cross-period comparisons on the LinkedIn-only subset of scraped data first, then check whether including Indeed changes results.

**Scraper query design effect:** Our scraper runs specific queries ("software engineer", "full stack engineer", etc.) in specific cities. The Kaggle dataset was collected with unknown queries and geographic scope. Different query strategies surface different jobs even on the same day. **Mitigation:** Document this as a limitation. Compare the title distributions to see if one dataset captures roles the other misses.

### 9c. Missing data audit

**Why a reviewer asks:** "With 76% salary missing in Kaggle and 96% missing from LinkedIn scrapes, any salary-based conclusion is drawn from a tiny, non-random subset."

**Produce a missing data table:**

| Field | Kaggle (n=~3,466 SWE) | Scraped LinkedIn (n=~10K SWE) | Scraped Indeed (n=~4K SWE) |
|---|---|---|---|
| Title | % | % | % |
| Description | % | % | % |
| Company name | % | % | % |
| Location | % | % | % |
| Seniority (native) | % | % | % |
| Salary (any) | % | % | % |
| Skills (structured) | % | % | % |
| Industry | % | % | % |
| Company size | % | % | % |
| Date posted | % | % | % |

**Missingness mechanism analysis (for salary):**

Per Hazell & Taska (2023), salary missingness is almost certainly MNAR (Missing Not At Random). Run these diagnostics:

1. **`missingno` heatmap**: `msno.heatmap(df)` — reveals whether salary missingness correlates with seniority, company size, industry. If correlations are strong, missingness is at least MAR.
2. **Salary missingness by seniority**: Compute the salary disclosure rate per seniority level. If entry-level discloses at a different rate than senior, salary-based seniority comparisons are biased.
3. **Salary missingness by company size**: Large companies may disclose differently than small ones.
4. **Salary missingness by platform**: Indeed (76%) vs. LinkedIn (4%) is a massive platform effect. Platform-provided salary estimates must be distinguished from employer-provided salaries.

**Decision:** Treat salary as a secondary analysis variable, not a primary one. All core analyses (seniority shifts, skill migration, structural break) should NOT depend on salary. Salary analysis is conditional on observability, with explicit caveat citing Hazell & Taska.

### 9d. Selection bias diagnostics

**Why a reviewer asks:** "Your data only captures jobs posted online, through specific queries, on specific platforms. How do you know your findings generalize?"

**The five selection mechanisms in our data (from research docs):**

1. **Platform selection**: LinkedIn overrepresents BA+ professional jobs. For SWE roles specifically, coverage is high (>80% posted online per Carnevale et al. 2014). But control occupations (nursing, civil engineering) have lower online posting rates — this confounds the DiD design.

2. **Algorithm selection**: LinkedIn's ranking algorithm optimizes for engagement, not representativeness. Promoted (paid) posts get 3-5x visibility. Our scraper captures what the algorithm surfaces, not a random sample.

3. **Scraper selection**: Our 28 queries × 20 cities × 25 results/query design creates deterministic gaps:
   - Max 25 results per query-city combo means we miss long-tail postings
   - 20 cities misses smaller metros entirely
   - Query tier design means roles that don't match any query are excluded

4. **Employer selection**: Staffing companies (Lensa, Dice, etc.) inflate some companies' representation. DataAnnotation has 168 postings in the Kaggle SWE set (5.4%) — likely an outlier.

5. **Temporal selection (volatility bias)**: Daily scraping oversamples longer-lived postings (Foerderer 2023). A job open for 60 days is 60x more likely to appear in any daily scrape than a 1-day posting. This biases toward hard-to-fill roles.

**Tests to run:**

| Test | What it checks | How |
|---|---|---|
| **Covariate balance (ASMD)** | Whether scraped data distributions match BLS benchmarks | Meta's `balance` package. ASMD < 0.1 is acceptable per Stuart et al. (2013) |
| **Company size distribution vs. BLS** | Whether we over-represent large firms | Compare our company size distribution against Census SUSB (Statistics of U.S. Businesses) for NAICS 5112 (Software Publishers) |
| **Geographic coverage map** | Whether our 20-city design biases results | Plot SWE postings by metro. Compare against OES metro-level SWE employment |
| **Query saturation check** | Whether 25 results/query is enough | For key queries, re-scrape with higher limits (50, 100) and compare distributions |
| **Posting duration analysis** | Whether we oversample long-lived postings | If `date_posted` is available, compute posting duration distribution. Compare against Foerderer (2023) benchmarks |

**Covariate balance protocol (from research docs):**

```python
from balance import Sample
sample = Sample.from_frame(scraped_df[['seniority', 'company_size', 'industry', 'state']])
target = Sample.from_frame(bls_benchmark_df[['seniority', 'company_size', 'industry', 'state']])
adjusted = sample.set_target(target).adjust()
# Reports ASMD per covariate — threshold: ASMD < 0.1
```

If ASMD > 0.1 for key covariates, apply inverse-probability-of-selection weighting (IPSW) to reweight our sample toward the BLS benchmark. Report results both with and without reweighting.

### 9e. Classifier validation

**Why a reviewer asks:** "Your entire study depends on correctly classifying jobs as SWE and correctly imputing seniority. How accurate are these classifiers?"

**SWE detection validation (3-tier):**

1. Sample 500 postings from each dataset: 250 classified as SWE + 250 classified as non-SWE (enriched with borderline titles containing "engineer", "developer", "software", "tech" that didn't match the pattern).
2. **Tier 2 (LLM):** Claude labels each as SWE / SWE-adjacent / non-SWE with reasoning. Prompt includes title, company, first 400 chars of description.
3. **Tier 3 (Human):** Annotator reviews LLM labels, correcting disagreements. Focus on borderline cases. Compute kappa between LLM and human.
4. Report precision, recall, F1 against the corrected gold standard.
5. **Known issues to check:**
   - Does "Software Development Engineer" match? (16 false negatives per day currently.)
   - Does "Product Engineer" match in both datasets consistently?
   - Does "Language Engineer, Artificial General Intelligence" count as SWE? (33 occurrences in non-SWE file.)

**Seniority classifier validation (3-tier):**

1. Sample 500 postings stratified by: source (Kaggle vs. scraped), predicted seniority (entry/mid/senior), and title ambiguity.
2. **Tier 2 (LLM):** Claude Sonnet pre-labels seniority from title + description with reasoning. Higher-quality model used here because seniority is a primary analysis variable.
3. **Tier 3 (Human):** Annotator corrects LLM labels. Compute kappa between LLM and human (target ≥ 0.80). Adjudicate disagreements.
4. Evaluate our imputer against the corrected gold standard. Report per-class precision/recall/F1.
5. **Critical check:** Run the same classifier on Kaggle data where LinkedIn native labels exist. Compare our imputation against LinkedIn's labels. If our classifier agrees with LinkedIn at a different rate for different seniority levels, that differential error biases cross-period comparisons.

**Classifier temporal stability:**

Our seniority classifier was designed from 2026 posting conventions. It may perform differently on 2024 data if title conventions changed. **Test:** Compute per-class accuracy on Kaggle (where native labels are available) and on scraped LinkedIn (where native labels are available). If accuracy differs significantly between periods, the classifier introduces a temporal artifact.

### 9f. Distribution comparisons for key analysis variables

**Why a reviewer asks:** "Before I believe your cross-period findings, show me the raw distributions. Are you comparing normal distributions? Skewed? Bimodal?"

For each key variable, produce distribution plots (histograms or KDEs) side by side for Kaggle vs. scraped, and run formal distribution comparison tests:

| Variable | Test | Why |
|---|---|---|
| Description length (chars) | KS test + QQ plot | RQ1 scope inflation proxy — must rule out scraping artifact |
| Description length after boilerplate removal | KS test + QQ plot | The apples-to-apples version |
| Seniority distribution | Chi-squared test | RQ1 core metric |
| Salary (where available) | KS test | Wage trends |
| Company size | KS test | Composition control |
| Word count of requirements section only | KS test | More targeted scope inflation measure than full description |
| Number of distinct skills mentioned | KS test | Skill breadth index |
| Years of experience required | KS test | Direct seniority requirement measure |
| Remote work rate | Proportion test | Compositional difference |

**Interpretation framework:** A statistically significant difference is NOT automatically evidence of labor market change. It could also indicate:
- Scraping method differences
- Platform changes
- Company composition differences
- Seasonal variation

For each significant difference, attempt to decompose it: how much is explained by composition (different companies, industries, geographies) vs. within-composition change?

### 9g. Compositional analysis: Is the comparison apples-to-apples?

**Why a reviewer asks:** "Maybe the seniority distribution shifted not because junior jobs disappeared, but because your 2026 scraper happened to capture a different set of companies than the 2024 Kaggle dataset."

**Decomposition approach:**

1. **Company overlap analysis:** Identify companies appearing in both datasets. For the overlapping set, compare seniority distributions. If the shift holds within the same companies, it's not a composition effect.

2. **Industry-controlled comparison:** Compare seniority distributions within matched industries (e.g., SWE postings in "Technology/Information" only, excluding healthcare SWE, finance SWE, etc.).

3. **Geography-controlled comparison:** Compare within matched metros (e.g., San Francisco SWE only, NYC SWE only).

4. **Company-size-controlled comparison:** Compare within matched size bands (e.g., large companies >10K employees only).

5. **Oaxaca-Blinder decomposition** (if warranted): Formally decompose the cross-period difference in any outcome (seniority share, skill prevalence) into:
   - A composition effect (different mix of companies/industries/geographies)
   - A within-composition effect (same companies posting differently)
   This is the gold standard in labor economics for separating "who's hiring" from "what they're hiring for."

### 9h. Company concentration analysis and normalization

**Why a reviewer asks:** "Your scraped data top-5 companies account for 30% of SWE postings. How do you know your findings aren't driven by the hiring patterns of a handful of large employers?"

**The problem:** Company concentration differs between datasets (Kaggle top-5 = 12.6%, scraped top-5 = 30.2%). A company like DataAnnotation (168 Kaggle SWE postings, 5.4%) may have unusual seniority distributions, skill requirements, or description styles that skew aggregate metrics. If one dataset is dominated by a few companies that the other doesn't have, cross-period differences may reflect company composition, not labor market change.

**Diagnostic steps:**

1. **Compute concentration metrics per dataset:**
   - Herfindahl-Hirschman Index (HHI): Sum of squared posting-share per company. HHI > 0.15 = moderately concentrated, > 0.25 = highly concentrated.
   - Top-5 / top-10 / top-20 company share of total SWE postings
   - Gini coefficient of company posting counts

2. **Identify dominant companies and audit them:**
   - For any company with >3% of SWE postings in either dataset, LLM reviews 20 postings per company. Prompt: "Is this a real software engineering job or crowdwork/data-annotation/template? Does it use a standardized template? What seniority level does the content suggest?" Human reviews any company the LLM flags as suspicious.
   - Flag companies that are functionally aggregators or crowdwork platforms even if not in the AGGREGATORS list

3. **Within-company vs. between-company decomposition:**
   - For companies appearing in BOTH datasets, compare their seniority distributions across periods. If the shift holds within the same companies, it's not a composition effect.
   - Compute the cross-period seniority shift (a) on the full sample, (b) on the overlapping-companies-only sample, and (c) on the non-overlapping sample. If (b) shows the same shift as (a), company composition is not driving the finding.

4. **Company-capped analysis (sensitivity):**
   - Cap each company at N postings (e.g., N = 10 or N = median company count) to prevent any single company from dominating.
   - Re-run key analyses on the capped sample. If findings hold, they're not driven by a few prolific posters.

5. **Company-level fixed effects (for regression analyses):**
   - Include company fixed effects in any regression model. This absorbs all between-company variation and isolates within-company changes over time.
   - Only works for companies appearing in both periods — report the overlap rate.

6. **Exclusion sensitivity tests:**
   - Re-run analyses excluding the top-5 companies from each dataset
   - Re-run excluding all aggregators/staffing companies
   - Re-run excluding DataAnnotation specifically (suspected crowdwork platform)

**Reporting:** Include a company concentration table in the methodology section:

| Metric | Kaggle SWE | Scraped SWE |
|---|---|---|
| Unique companies | X | X |
| Top-1 company share | X% | X% |
| Top-5 share | 12.6% | 30.2% |
| Top-10 share | X% | X% |
| HHI | X | X |
| Company overlap (Jaccard) | X | |
| Companies in both datasets | X | |

### 9i. Temporal stability and seasonality checks

**Why a reviewer asks:** "With only two snapshots 23 months apart, how do you separate genuine structural change from normal seasonal or cyclical variation?"

**Checks:**

1. **JOLTS seasonal pattern:** Plot JOLTS information sector openings by month. Show that March-to-April variation is small relative to the cross-year change we observe. If BLS data shows a typical March-April seasonal swing of ±5% but we observe a 20% shift in junior share, seasonality alone cannot explain it.

2. **Within-period stability (scraped data):** We have 14 daily scrapes. Compute daily seniority distributions and test for day-to-day stability. If the distribution is stable within our 2-week window, it's unlikely that a 1-month seasonal offset (March vs. April) drives our findings.

3. **Kaggle within-snapshot variation:** The Kaggle data covers ~4 weeks (March 24 - April 20). Check whether postings from early vs. late in this window differ. If they don't, within-month variation is negligible.

4. **External triangulation:** Compare our findings against Revelio Labs trends (SOC 15 hiring, openings, salary) which have monthly resolution across 2021-2026. Do Revelio trends show a gradual decline or a discrete break? Does the slope accelerate around late 2025?

### 9j. Power analysis: Do we have enough data?

**Why a reviewer asks:** "With only ~3,466 Kaggle SWE postings, do you have statistical power to detect a meaningful difference?"

**Key power calculations needed:**

| Analysis | Effect size to detect | Sample sizes | Estimated power |
|---|---|---|---|
| Junior share change (chi-squared) | 5pp shift (e.g., 12% → 7%) | 3,466 vs. 14,391 | Compute |
| Description length change (Mann-Whitney) | Cohen's d = 0.2 (small) | 385 entry-level Kaggle vs. N entry-level scraped | Compute |
| Skill prevalence change (proportion test) | 5pp shift in skill mention rate | Same | Compute |
| DiD (SWE vs. control) | Interaction effect | SWE: ~17K, Control: ~1K-3K | Compute — control sample may be binding constraint |

**Kaggle entry-level SWE is the binding constraint:** Only 385 Kaggle SWE postings are labeled "entry level" (native). After imputation, this might rise to 500-700, but it's still small. Compute the minimum detectable effect size given this sample.

**Control occupation sample size:** On March 18, the non-SWE file had 874 control-pattern matches (nurses, civil/mechanical/electrical/chemical engineers). Across 14 days, we may have ~3,000-5,000 unique control postings. But the Kaggle control count depends on the raw Kaggle data (not just SWE-filtered). We need to check this.

### 9k. Robustness pre-registration: What specifications will we test?

**Why a reviewer asks:** "You could have tried 50 different specifications and reported the one that worked. How do I know you didn't?"

Per the research docs, define the specification space BEFORE looking at results:

**SWE definition variants:**
1. Narrow: Current `SWE_PATTERN` (core SWE titles only)
2. Broad: Add "data scientist", "data analyst", "product engineer"
3. Excluding adjacent: Remove "data engineer", "ML engineer" (these may have different dynamics)

**Seniority classification variants:**
1. Our imputer applied to all rows (recommended default)
2. LinkedIn native labels where available, imputed where missing
3. Description-only classifier (ignore titles)

**Dedup variants:**
1. Strict: Exact match on (title, company, location)
2. Standard: Near-dedup with similarity ≥ 0.70
3. Loose: Near-dedup with similarity ≥ 0.50

**Sample variants:**
1. Full sample
2. LinkedIn only (excludes Indeed composition effect)
3. Excluding aggregator postings
4. Top-10 metros only (more comparable geographic scope)
5. Excluding top-5 most common companies (reduces concentration bias)

**Key findings must hold across all defensible specifications.** Use the `specification_curve` package to visualize this. If a finding is fragile (holds under some specifications but not others), it is reported as suggestive, not conclusive.

### 9l. Placebo and falsification tests (pre-registration)

**Note:** This section defines the placebo tests. Stage 12b-12c executes them after Stage 11 produces results.

**Why a reviewer asks:** "Maybe your method finds 'structural change' in any two snapshots, regardless of whether anything actually changed."

**Placebos to pre-register:**

1. **Control occupation placebo:** Run the same seniority-shift analysis on control occupations (civil engineering, nursing, mechanical engineering). If they show the same "structural change" as SWE, the finding is confounded by macro trends or measurement artifacts, not AI-specific restructuring.

2. **Within-Kaggle time-split placebo:** Split Kaggle into early April (before Apr 12, n=30,101) vs. late April (Apr 12+, n=93,748). SWE counts: 743 vs. 2,723. Run the same analysis across the two halves. If we find a "shift" within a 2-week window, our method is detecting noise.

3. **Shuffled-label test:** Randomly permute the dataset labels (Kaggle vs. scraped) and re-run the analysis 10,000 times. The observed effect size should exceed 95% of permuted effect sizes.

4. **Within-scraped week-over-week:** Split 14 days into week 1 (Mar 5-11) vs. week 2 (Mar 12-18). Run the same analyses. Expect null results.

4. **Null-effect occupations:** Identify occupations with no theoretical reason to be affected by AI coding agents (e.g., registered nurses, civil engineers). Run the full analysis pipeline on these as negative controls. Expect null results.

### 9m. Bias threat summary table

Produce a consolidated table for the methodology section:

| Bias | Direction | Magnitude estimate | Mitigation | Residual risk |
|---|---|---|---|---|
| Platform selection (LinkedIn overrepresents tech/professional) | Favors SWE coverage; underrepresents control occupations | ~11pp for tech occupations (Hershbein & Kahn) | Post-stratification against OES | Low for SWE; moderate for controls |
| Algorithm selection (promoted posts) | Unknown direction | Unknown | Cannot correct; acknowledge | Moderate |
| Scraper query design (25 results × 20 cities) | Misses long-tail postings | Unknown | Query saturation check | Moderate |
| Aggregator contamination | Inflates some company counts; adds boilerplate | 9% of scraped, ~15% of Kaggle SWE | Flag and sensitivity analysis | Low after flagging |
| Temporal selection (volatility bias) | Oversamples long-lived postings | 60:1 for 60-day vs. 1-day postings (Foerderer) | Report duration distribution; consider IPW | Moderate |
| Kaggle provenance unknown | Could bias anything | Unknown | Treat as stated limitation | High (irreducible) |
| Ghost jobs | Inflates entry-level tech postings | 18-27% of all postings (CRS 2025) | Flag and sensitivity analysis | Moderate |
| Salary missingness (MNAR) | Biases salary-based analyses | 72% missing in Kaggle, 96% missing in LinkedIn scrape | Drop-and-flag (Azar et al. 2022) | High for salary outcomes; N/A for non-salary |
| Platform changes (2024 → 2026) | Could create artificial differences | Unknown | Investigate LinkedIn changelog; run LinkedIn-only comparison | Moderate (irreducible) |
| Company composition shift | Could drive apparent seniority shift | Unknown until tested | Oaxaca-Blinder decomposition; within-company comparison | Low after decomposition |
| Seasonal offset (April vs. March) | Could inflate/deflate metrics | Typically small for adjacent months | JOLTS seasonal comparison | Low |

---


---

## Spot-check protocol

Manual review is non-negotiable for publication quality. The following spot-checks should be conducted after the pipeline runs:

All spot-checks use the 3-tier review protocol (rules → LLM → human). See `plan-preprocessing.md` for the full spot-check table with tier assignments and sample sizes.

---

---

## Stage 10: Exploration & discovery

This phase builds intuition about the data before formal hypothesis testing. Every exploration here is designed to either (a) generate visualizations and tables that go directly into the paper, (b) surface unexpected patterns that refine our research questions, or (c) validate that the preprocessed data behaves as expected before committing to expensive analyses.

**Key tool choices for this stage:**

| Tool | What it does | Why we use it here |
|---|---|---|
| [**JobBERT-v2**](https://huggingface.co/TechWolf/JobBERT-v2) | Sentence transformer fine-tuned on 5.5M job title-skill pairs (MPNet base, 1024d). **Max 64 tokens — titles only.** | Use for **title-level** tasks: classification, title clustering, title similarity. NOT for full descriptions (64-token limit). See Stage 5d in `plan-preprocessing.md` for model validation. |
| **Description model** (selected via Stage 5d benchmark) | General-purpose sentence transformer (`all-mpnet-base-v2` or `e5-large-v2`). 384-512 token context. | Use for **description-level** tasks: topic modeling (BERTopic), content convergence, drift measurement, embedding space exploration. |
| [**BERTopic**](https://bertopic.com/) | Neural topic modeling (embedding → UMAP → HDBSCAN → c-TF-IDF). Supports dynamic topic modeling over time. | Discovers emergent skill clusters we didn't think to look for. Handles semantic meaning, not just keywords. |
| [**BERTrend**](https://github.com/rte-france/BERTrend) | Runs BERTopic per time slice, merges across windows, classifies topics as noise / weak signal / strong signal. | Detects genuinely new topics (e.g., AI-orchestration skills) that emerge between 2024 and 2026. Standard BERTopic assumes a fixed topic set — BERTrend relaxes this. |
| [**Fightin' Words**](https://github.com/Wigder/fightin_words) | Log-odds-ratio with Dirichlet prior for pairwise corpus comparison. Produces both effect size and z-score per word. | The statistically rigorous way to answer "what words distinguish corpus A from corpus B?" Handles corpus size imbalance (our datasets are 3K vs. 14K). |
| [**Scattertext**](https://github.com/JasonKessler/scattertext) | Interactive HTML visualization of distinguishing terms between two corpora. | Produces publication-quality figures and lets us visually inspect what drives corpus differences. |
| [**KeyBERT**](https://github.com/MaartenGr/KeyBERT) | Keyword/keyphrase extraction using BERT embeddings + cosine similarity. | Extracts the most representative terms from each posting or group of postings without a predefined dictionary. |
| [**ESCO Skill Extractor**](https://github.com/nestauk/ojd_daps_skills) | Maps free-text skill mentions to the ESCO/Lightcast Open Skills taxonomy. | Structured skill extraction grounded in an international standard. Gives us a controlled vocabulary instead of raw keywords. |
| [**Nesta OJD Skills Library**](https://github.com/nestauk/ojd_daps_skills) | End-to-end pipeline: extract skill phrases from job ads, map to ESCO or Lightcast taxonomy. | Built specifically for job ad analysis. Handles the full pipeline from raw text to structured skill tags. |

### 10a. Raw data inspection (3-tier: rules → LLM → human)

**Before any automated analysis, look at the data.** Automated methods can produce plausible-looking results from garbage data. We use the 3-tier review protocol (defined in `plan-preprocessing.md`) to scale inspection beyond what manual review alone allows.

**Tier 1 — Rule-based screening (full dataset):**
- Flag descriptions < 100 chars, > 15,000 chars, or with non-ASCII majority
- Flag postings where seniority label contradicts years-of-experience in description
- Flag entry-level titles with > $150K salary or 5+ years required
- Flag company name mismatches (aggregator name vs. description employer)
- Output: CSV of flagged rows with flag reasons

**Tier 2 — LLM bulk review (500 postings):**
Sample 500 postings (stratified: 150 Kaggle SWE, 150 scraped SWE, 100 non-SWE, 100 extreme values from Tier 1 flags). For each, run:

```bash
claude -p "Review this job posting for data quality issues.

Title: ${TITLE}
Company: ${COMPANY}
Seniority label: ${SENIORITY}
Description (first 800 chars): ${DESC}

Assess:
1. Is this a real job posting? (yes/spam/template-only/boilerplate-only)
2. Does the seniority label match the description content? (match/mismatch/unclear)
3. Is the description complete? (full/truncated/mostly-boilerplate)
4. Are the skill requirements realistic for the stated seniority? (yes/inflated/deflated)
5. Any other quality concerns? (ghost-job/aggregator-repost/non-english/other/none)
6. One-sentence summary of the role.

Respond as JSON with fields: is_real, seniority_match, desc_quality, requirements_realistic, concerns, summary" --output-format json
```

- Output: JSONL file with LLM assessments for all 500 postings
- Aggregate stats: % flagged per issue type, by dataset

**Tier 3 — Human review (targeted, ~50 postings):**
- Review all postings LLM flagged as problematic (spam, mismatch, ghost-job)
- Review 10 random "clean" postings to validate LLM accuracy
- **Side-by-side comparison (10 matched pairs):** Find 10 companies in both datasets. Pull one SWE posting from each period. Compare descriptions: structure changes, requirement changes, AI mentions. Quote specific examples in the paper.

**Output:** `data/quality_review.md` — annotated examples tagged with `[boilerplate]`, `[ghost-job?]`, `[skill-inflation]`, `[good-example]`, `[misclassified]`. Plus `data/llm_review_results.jsonl` for the full LLM assessment dataset.

### 10b. Embedding space exploration

**Goal:** Visualize how job postings cluster in semantic space and whether clusters align with our seniority/occupation categories.

**Implementation:**

1. **Embed all SWE postings** using the description-level model selected in Stage 5d (likely `all-mpnet-base-v2` or `e5-large-v2`). Use `title + first 400 words of description_core` as input. For title-only tasks (clustering by job title, SWE classification), use JobBERT-v2 separately.

   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("TechWolf/JobBERT-v2")

   # Embed title + first 200 words of description_core
   texts = (df['title'] + ' ' + df['description_core'].str[:800]).tolist()
   embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
   ```

2. **UMAP projection to 2D:**
   ```python
   import umap
   reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine')
   coords = reducer.fit_transform(embeddings)
   ```

3. **Visualization layers** (produce multiple views of the same embedding space):
   - Color by **seniority** (junior/mid/senior) — do seniority levels form distinct clusters, or do they overlap? Overlap = the seniority boundary is fuzzy in practice.
   - Color by **period** (2024 vs. 2026) — do the two periods occupy different regions, or are they interleaved? Separation = real distributional shift. Interleaving = similar posting content across periods.
   - Color by **company** (top-10 companies highlighted) — do individual companies form tight clusters (standardized templates) or spread across the space?
   - Color by **topic** (from BERTopic, step 10d) — visual confirmation that topic clusters are semantically coherent.

4. **Quantitative embedding analysis:**
   - **Junior-senior centroid distance:** Compute the cosine distance between the average junior embedding and the average senior embedding, separately for each period. If this distance shrank between 2024 and 2026, junior roles are converging toward senior roles in content (RQ1 evidence).
   - **Within-seniority variance:** Compute the average pairwise cosine distance within each seniority level. If junior postings have higher variance in 2026 than 2024, the category is becoming more heterogeneous (redefinition, not disappearance).
   - **Cross-period drift per seniority:** For each seniority level, compute the cosine distance between the 2024 centroid and the 2026 centroid. If junior roles drifted more than senior roles, that's directional evidence for RQ1.

### 10c. Corpus comparison — Fightin' Words & Scattertext

**Goal:** Identify the specific words and phrases that statistically distinguish different groups (junior vs. senior, 2024 vs. 2026, SWE vs. control).

**Comparisons to run (6 total):**

| Comparison | What it tests | RQ |
|---|---|---|
| Junior 2024 vs. Junior 2026 | How junior role language changed | RQ1 |
| Senior 2024 vs. Senior 2026 | How senior role language changed | RQ6 |
| Junior 2024 vs. Senior 2024 | Baseline junior-senior gap | RQ1, RQ2 |
| Junior 2026 vs. Senior 2026 | Current junior-senior gap | RQ1, RQ2 |
| Junior 2026 vs. Senior 2024 | Are 2026 juniors linguistically similar to 2024 seniors? | RQ1 (redefinition hypothesis) |
| SWE 2026 vs. Control 2026 | SWE-specific vs. general labor market language | RQ4 |

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
   ```python
   import scattertext as st
   corpus = st.CorpusFromPandas(df, category_col='period',
                                 text_col='description_core',
                                 nlp=nlp).build()
   html = st.produce_scattertext_explorer(corpus,
       category='2026-03', category_name='March 2026',
       not_category_name='April 2024')
   open('scattertext_2024_vs_2026.html', 'w').write(html)
   ```
   Produces an interactive HTML plot. Each dot is a word. Position = relative frequency in each corpus. Hover for context. This is both an analysis tool and a publication figure.

3. **What to look for:**
   - **AI/LLM terms emerging:** Do "LLM", "AI agent", "prompt engineering", "RAG", "vector database" appear in 2026 but not 2024?
   - **Management terms declining in senior roles:** Do "mentorship", "coaching", "team leadership", "hiring" shift downward in 2026 senior postings? (RQ6)
   - **Senior terms appearing in junior roles:** Do "system design", "architecture", "end-to-end", "cross-functional" appear more in 2026 junior postings than 2024? (RQ1 skill migration)
   - **Years-of-experience inflation:** Do "3+ years", "5+ years" appear more frequently in 2026 entry-level postings? (ghost job / scope inflation signal)

### 10d. Topic discovery — BERTopic & BERTrend

**Goal:** Discover latent topic structure and track which topics are emerging, stable, or declining between periods. This catches skill categories and role archetypes that our predefined keyword lists miss.

**Implementation:**

1. **Base BERTopic on full SWE corpus (both periods combined):**

   ```python
   from bertopic import BERTopic
   from sentence_transformers import SentenceTransformer

   embedding_model = SentenceTransformer("TechWolf/JobBERT-v2")
   topic_model = BERTopic(
       embedding_model=embedding_model,
       min_topic_size=30,          # ~0.2% of SWE corpus
       nr_topics="auto",
       verbose=True
   )
   topics, probs = topic_model.fit_transform(descriptions)
   ```

   **Critical:** Use `description_core` (boilerplate-stripped). EEO statements and benefits sections are the #1 source of spurious topics. Our research docs flag this: "Remove EEO statements, benefits sections, company boilerplate. This is the single highest-impact step."

2. **Dynamic topic modeling (topics over time):**
   ```python
   topics_over_time = topic_model.topics_over_time(
       descriptions, timestamps=dates, nr_bins=6  # e.g., monthly or bimonthly
   )
   topic_model.visualize_topics_over_time(topics_over_time)
   ```
   This shows which topics are growing and shrinking. With only 2 time points (April 2024, March 2026), the resolution is limited — but the signal we're looking for (new AI topics, declining management topics) should be detectable even with 2 bins.

3. **BERTrend for emerging signal detection:**

   BERTrend goes beyond standard BERTopic by classifying each topic as **noise**, **weak signal**, or **strong signal** based on popularity trends. This is designed for exactly our use case: detecting whether AI-orchestration skills are a weak signal in 2024 that became a strong signal by 2026.

   ```python
   from bertrend import BERTrend
   # Process data in time-sliced batches
   bertrend = BERTrend(embedding_model="TechWolf/JobBERT-v2")
   bertrend.fit(docs_by_period)  # dict of {period: [docs]}
   signals = bertrend.get_signals()  # weak, strong, noise classification
   ```

4. **Topic-level analysis for each RQ:**

   | RQ | What to look for in topics | Expected signal |
   |---|---|---|
   | RQ1 | Topics that appear in senior postings in 2024 but migrate to junior postings in 2026 | Topics like "system design", "architecture decisions" |
   | RQ2 | The temporal order in which skill-topics appear in junior postings | Sequence: cloud → CI/CD → system design → AI tools |
   | RQ6 | Topics in senior postings: management-heavy in 2024, AI-heavy in 2026 | "Team leadership, hiring" declining; "AI integration, agent orchestration" emerging |
   | RQ4 | Whether the same topic shifts appear in control occupations | They should NOT (if they do, it's macro confounding) |

5. **Guided BERTopic** (semi-supervised):

   BERTopic supports [guided topic modeling](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html) where you provide seed topics. Use our RQ-derived skill categories as seeds:
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
   This nudges the model to discover topics aligned with our hypotheses while still allowing it to find unexpected topics.

### 10e. Structured skill extraction

**Goal:** Extract structured, taxonomy-mapped skills from free-text descriptions. This goes far beyond keyword matching — it handles synonyms, multi-word phrases, and maps to a standard vocabulary.

**Why this matters:** Our current skill analysis uses a hand-coded list of 16 keywords. This misses skills we didn't think of, conflates different uses of the same word, and can't handle synonyms ("k8s" = "Kubernetes" = "container orchestration").

**Implementation — two complementary approaches:**

1. **Nesta OJD Skills Library** (taxonomy-mapped):
   ```python
   from ojd_daps_skills.pipeline.skill_ner import SkillNER
   from ojd_daps_skills.pipeline.skill_match import SkillMapper

   ner = SkillNER()
   mapper = SkillMapper(taxonomy="esco")  # or "lightcast"

   skills_raw = ner.extract(description_core)
   skills_mapped = mapper.map(skills_raw)
   # Returns: [{"skill": "Kubernetes", "esco_code": "S4.8.1", "confidence": 0.92}, ...]
   ```

   Maps to ESCO (13,890 skills) or Lightcast Open Skills taxonomy. Gives us a controlled vocabulary for cross-period comparison — "Docker" and "containerization" both map to the same ESCO skill code.

2. **KeyBERT** (unsupervised, catches what taxonomies miss):
   ```python
   from keybert import KeyBERT
   kw_model = KeyBERT(model="TechWolf/JobBERT-v2")

   keywords = kw_model.extract_keywords(
       description_core,
       keyphrase_ngram_range=(1, 3),
       stop_words='english',
       top_n=15
   )
   # Returns: [("distributed systems", 0.72), ("react native", 0.68), ...]
   ```

   Using JobBERT-v2 as the backbone means keyphrases are scored by relevance to the job domain, not just general text statistics. This catches emerging terms (e.g., "agentic workflow", "vibe coding") that aren't in any taxonomy yet.

3. **Skill prevalence analysis (replaces hand-coded keyword lists):**
   - For each ESCO-mapped skill, compute prevalence by seniority × period
   - Identify skills with the largest prevalence change between periods
   - Identify skills that migrated from senior-only to junior+senior
   - Produce the task migration map (RQ2) grounded in a standard taxonomy rather than researcher-selected keywords

### 10f. Seniority boundary analysis

**Goal:** Understand where the junior/senior boundary actually lies in the data, and whether it moved between periods.

**This addresses the core of RQ1:** Are junior roles being redefined? If the content of 2026 "junior" postings looks like 2024 "mid-senior" postings, that's direct evidence of redefinition.

**Implementation:**

1. **Embedding-based seniority boundary:**
   - Train a simple logistic regression on JobBERT-v2 embeddings to predict seniority (junior vs. senior) using 2024 data only
   - Apply this classifier to 2026 data. If 2026 "junior" postings are classified as "senior" by the 2024 model at a higher rate than 2024 "junior" postings, the boundary has shifted. This is the "content convergence" metric from the validation plan.
   - Report: % of 2026 junior postings that the 2024-trained model classifies as senior (the "redefinition rate")

2. **Decision boundary visualization:**
   - In the UMAP space from 10b, draw the decision boundary of the 2024-trained seniority classifier
   - Overlay 2026 junior postings. How many fall on the "senior" side of the 2024 boundary?
   - This is a powerful visual for the paper: it literally shows junior postings crossing into senior territory.

3. **Feature importance for the boundary:**
   - What words/skills most strongly predict "senior" vs. "junior" in the 2024 model?
   - Which of those features are now present in 2026 junior postings?
   - Use SHAP values or logistic regression coefficients for interpretability.

### 10g. Requirements section parsing

**Goal:** Extract structured data from the requirements/qualifications section of job descriptions, rather than relying on the full description text.

**Why this is separate from full-text analysis:** The requirements section is the most decision-relevant part of a job posting for applicants and for our research. "Scope inflation" (RQ1) should be measured by what's required, not what's in the "About Us" section.

**Implementation:**

1. **Section extraction** (from Stage 3 boilerplate removal):
   - Parse `description_core` into sections: responsibilities, requirements/qualifications, nice-to-haves
   - Analyze requirements sections separately from full descriptions

2. **Structured requirement extraction:**
   - **Years of experience:** Extract all "X+ years" patterns. Compute min, max, and median years required per seniority level × period. Test for inflation.
   - **Education requirements:** Extract degree mentions (BS, MS, PhD). Compute degree distribution per seniority × period.
   - **Technology requirements:** Count distinct technologies mentioned in requirements (not just description). Are junior roles in 2026 requiring more technologies than in 2024?
   - **Soft skill requirements:** Extract management/leadership/communication mentions from requirements specifically. Are junior roles now requiring "cross-functional collaboration" and "stakeholder management"?

3. **Requirement count as scope metric:**
   - Count the number of bullet points or distinct requirements per posting
   - Compare this count across seniority × period
   - This is a more targeted "scope inflation" metric than description word count, which is confounded by boilerplate

### 10h. Temporal drift measurement

**Goal:** Quantify how much the posting landscape changed between April 2024 and March 2026, and characterize the direction of change.

**Implementation:**

1. **Corpus-level embedding drift:**
   ```python
   # Compute centroid embeddings for each period
   centroid_2024 = embeddings_2024.mean(axis=0)
   centroid_2026 = embeddings_2026.mean(axis=0)
   drift = 1 - cosine_similarity([centroid_2024], [centroid_2026])[0][0]
   ```
   Do this overall, and per seniority level. Report as a table.

2. **Vocabulary drift (JSD):**
   - Build unigram frequency distributions for each period
   - Compute Jensen-Shannon divergence between them
   - Do this for the full SWE corpus, and separately for junior-only and senior-only
   - JSD for junior postings vs. JSD for senior postings: if junior postings changed more than senior, that's RQ1 evidence

3. **Nearest-neighbor stability:**
   - For each posting in the overlap set (companies appearing in both periods), find its k=10 nearest neighbors
   - What fraction of neighbors are from the same period vs. the other period?
   - If 2026 junior postings' nearest neighbors are mostly 2024 senior postings, that's strong evidence of content convergence

4. **Keyword emergence/disappearance:**
   - Terms appearing in 2026 at >1% prevalence but absent from 2024 (or <0.1%): these are emerging requirements
   - Terms appearing in 2024 at >1% prevalence but absent from 2026: these are declining requirements
   - Do this separately for junior and senior postings

### 10i. Company-level patterns

**Goal:** Understand whether observed shifts are driven by within-company changes (same companies posting differently) or between-company changes (different companies dominating).

**Implementation:**

1. **Company overlap set analysis:**
   - Identify companies appearing in both Kaggle and scraped datasets
   - For overlapping companies: compare their seniority distributions, skill mentions, description length across periods
   - This is the within-company change signal — not confounded by composition

2. **Company archetypes (via clustering):**
   - Cluster companies by their posting profiles (average embedding, seniority mix, skill distribution)
   - Are there distinct "types" of SWE employers? (e.g., FAANG-style, startup-style, consulting-style, government)
   - Did the relative share of these archetypes change between periods?

3. **Firm-size effects:**
   - Split by company size bands (1-50, 50-500, 500-5000, 5000+)
   - Run all key metrics (seniority distribution, skill prevalence, description length) within each size band
   - Does scope inflation show equally in startups and large enterprises, or is it concentrated?

### 10j. Ghost job and anomaly profiling

**Goal:** Characterize the ghost job phenomenon and understand whether it biases our findings.

**Implementation:**

1. **Ghost job feature analysis:**
   - For postings flagged as ghost-risk in Stage 8, profile them: Which companies? Which seniority levels? Which geographies? How do their descriptions differ from non-ghost postings?
   - Do ghost jobs have systematically different skill requirements (more inflated)?

2. **Anomaly detection:**
   - Use isolation forest or DBSCAN on the embedding space to identify outlier postings
   - Manually review the outliers: spam, duplicate templates, non-English, non-US, misclassified occupation?
   - Report the anomaly rate and exclude from analysis with sensitivity check

### 10k. Cross-occupation comparison

**Goal:** Establish that observed changes are SWE-specific, not part of a broader labor market trend (RQ4 groundwork).

**Implementation:**

1. **Run the same exploration on control occupations:**
   - Embedding space, Fightin' Words, topic modeling, skill extraction — all on civil engineering, nursing, mechanical engineering postings
   - The key question: do control occupations show the same patterns (AI skill emergence, scope inflation, seniority compression)?
   - If they do, our SWE findings are confounded. If they don't, we have DiD evidence.

2. **SWE-adjacent occupation analysis:**
   - Data scientist, product manager, UX designer — these are AI-exposed but not SWE
   - Do they show similar restructuring patterns? This tests whether the effect is specific to coding or broader to tech

3. **Cross-occupation embedding distance:**
   - How far apart are SWE, SWE-adjacent, and control postings in embedding space?
   - Is the distance between SWE and SWE-adjacent shrinking (role convergence)?

### Exploration outputs

The exploration phase produces:

| Output | Type | Used in |
|---|---|---|
| Annotated raw sample | Markdown file | Qualitative examples for paper |
| UMAP embedding plots (4 color schemes) | PNG + interactive HTML | Paper figures |
| Scattertext comparisons (6 pairs) | Interactive HTML | Paper figures, appendix |
| Fightin' Words tables (6 comparisons) | CSV + sorted tables | Paper tables |
| BERTopic model + topic list | Saved model + CSV | RQ2 analysis input |
| BERTrend signal report | CSV (topic × signal strength) | RQ2, RQ3 analysis input |
| ESCO-mapped skill prevalence table | Parquet | RQ2 task migration map |
| KeyBERT emerging terms list | CSV | Appendix |
| Seniority boundary classifier | Saved model | RQ1 redefinition rate |
| Requirements section structured data | Parquet (years, degree, tech count) | RQ1, RQ2 |
| Company-level metric table | Parquet | Compositional analysis |
| Ghost job profile | Markdown + CSV | Methodology section |

---


---

## How validation feeds into the methodology section

The validation battery produces the following tables/figures for the paper:

| Output | Section | Content |
|---|---|---|
| **Table: Data sources and coverage** | §3.1 Data | Date ranges, sample sizes, platform, selection mechanism |
| **Table: Missing data rates** | §3.1 Data | Per-field missingness by dataset (from 9c) |
| **Table: Representativeness** | §3.2 Validation | Our distribution vs. OES, with correlations and dissimilarity indices (from 9a) |
| **Table: Classifier performance** | §3.3 Classification | Per-class precision/recall/F1 for SWE detection and seniority imputation (from 9e) |
| **Table: Cross-dataset comparability** | §3.2 Validation | Distribution comparison tests for key variables (from 9f) |
| **Table: Bias threat summary** | §3.4 Limitations | The bias table from 9l |
| **Figure: Specification curve** | §4 Results | Effect stability across all defensible specifications (from 9j) |
| **Table: Placebo tests** | §4 Results | Null results on control occupations and random splits (from 9k) |
| **Table: Data funnel** | §3.1 Data | Raw → cleaned → final counts (from Stage 4f) |

---


---

## Implementation order

### Phase 2: Validation (after `unified.parquet` exists)

```
10. Representativeness checks (9a)           ← pull BLS/JOLTS benchmarks
11. Cross-dataset comparability (9b)         ← distribution comparisons
12. Missing data audit (9c)                  ← missingno diagnostics
13. Selection bias diagnostics (9d)          ← balance package, coverage analysis
14. Classifier validation (9e)               ← gold-standard annotation
15. Distribution comparisons (9f)            ← KS tests, histograms
16. Compositional analysis (9g)              ← company overlap, Oaxaca-Blinder
17. Company concentration (9h)               ← HHI, capping, within-company analysis
18. Temporal stability (9i)                  ← daily variance, JOLTS seasonality
19. Power analysis (9j)                      ← sample size calculations
20. Robustness specification space (9k)      ← define before looking at results
21. Placebo/falsification tests (9l)         ← control occupation null tests
```

Most checks (9a-9i) can run in parallel. 9k (specification space) should be defined before 9l (placebo tests).

### Phase 4: Exploration & Discovery

```
22. Raw data inspection & manual review (10a)
23. Embedding space exploration (10b)
24. Corpus comparison — Fightin' Words (10c)
25. Topic discovery — BERTopic + BERTrend (10d)
26. Structured skill extraction (10e)
27. Seniority boundary analysis (10f)
28. Requirements section parsing (10g)
29. Temporal drift measurement (10h)
30. Company-level patterns (10i)
31. Ghost job and anomaly profiling (10j)
32. Cross-occupation comparison (10k)
```

**Iteration:** Validation or exploration results may trigger re-runs of preprocessing (see `plan-preprocessing.md`).
