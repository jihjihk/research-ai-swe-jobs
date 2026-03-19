# Research: Representativeness Testing, Bias Detection, and Publication-Ready Data Validation

Date: 2026-03-18
Context: Supports RQ1-RQ4. Builds on prior research (research-data-normalization-job-postings.md) which covered normalization, dedup, seasonality, and reporting standards. This research focuses on the specific statistical tests, diagnostic procedures, and validation work needed to make our scraped data defensible for publication.

## Question

What data science procedures do we need to run on our scraped job posting data before it's rigorous enough for academic publication? What specific statistical tests, validation procedures, and bias diagnostics do published papers in labor economics and computational social science actually perform?

## Sub-questions explored

1. Representativeness diagnostics — statistical tests and benchmarks
2. Missing data analysis — handling MNAR salary, unlabeled seniority
3. Selection bias diagnostics — beyond post-stratification
4. Text classifier validation — seniority imputation, SWE detection accuracy
5. Robustness and sensitivity analysis — what reviewers expect

## Findings

### 1. Representativeness diagnostics

**What rigorous papers actually do:**

The canonical approach (Hershbein & Kahn 2018, AER) compares scraped data distributions against official sources at multiple granularities:

| Comparison | Their benchmark | Their result | Acceptable threshold |
|-----------|----------------|-------------|---------------------|
| State-level employment shares | OES | r = 0.98 | > 0.95 |
| Occupation x MSA wage levels | OES | r = 0.84 | > 0.80 |
| Industry shares | CPS | Within 3pp | Within 3-5pp |

**Specific tests used in the literature:**
- **Pearson correlation** of occupation/industry shares against official data (Hershbein & Kahn)
- **Dissimilarity index** (Duncan index): 0 = identical distributions, 100 = complete mismatch. Used by OECD (2024) for international benchmarking of Lightcast. No hard threshold, but report and discuss.
- **Chi-squared goodness-of-fit** (`scipy.stats.chisquare`): Tests whether scraped occupation distribution matches official distribution
- **KL divergence** (`scipy.stats.entropy`): Measures information-theoretic distance between distributions
- **KS test** (`scipy.stats.ks_2samp`): For continuous variables like salary
- **Time-series correlation**: Overlay scraped posting volume against JOLTS openings over time (Turrell et al. 2019 achieved r = 0.95 after correction, up from r = 0.65 raw)

**Common mistake to avoid:** Validating only in aggregate. Hershbein & Kahn found r = 0.98 at state level, but computer occupations are overrepresented by ~11 percentage points at the occupation level. Aggregate correlation gives false confidence. You must validate at the level of analysis — for us, that means SWE-related SOC codes specifically, not all occupations.

**Another trap:** Coverage ratios shift unevenly across occupations over time (OECD 2021). A 2023-2024 bias profile is not the same as a 2026 bias profile. This is the most dangerous confound for cross-temporal comparison. Validate each time period separately.

**How to pull benchmarks:**
- **FRED/JOLTS**: Use `fredapi` Python package. Key series: `JTSJOL` (total openings), `JTS5100JOL` (information sector). Free API key from FRED.
- **BLS OES**: Direct REST API at bls.gov/developers. Series IDs follow `OEUM[area][industry][occupation][datatype]` pattern.
- Compare your scraped occupation distribution against OES employment shares, and your posting volume against JOLTS levels.

**Sources**: Hershbein & Kahn (2018), AER; OECD (2024), LEED Paper; Turrell et al. (2019/2023), NBER/PMC; Carnevale et al. (2014), Georgetown CEW; Indeed Hiring Lab (2024).

### 2. Missing data analysis

**The salary problem is worse than it looks:**

Hazell & Taska (2023, NBER WP 31984) found only **17%** of Burning Glass postings include wage information. Since BG covers ~70% of US vacancies, wage-posting jobs represent roughly **10% of all vacancies**. The Kaggle dataset is ~70% missing on salary.

**Salary missingness is almost certainly MNAR (Missing Not At Random):**
- Firms strategically choose whether to post wages — high-wage firms and very low-wage firms are both more likely to omit
- Posted wages deviate from BLS benchmarks by **20-40%** depending on wage level
- Major job boards started adding **platform-imputed wage ranges** around 2018, inflating apparent wage coverage by ~520% — these are NOT employer-provided salaries and should be identified and flagged

**What published papers do:**

| Approach | When to use | Papers using it |
|---------|------------|----------------|
| **Drop and flag** | Default for salary. Analyze only wage-posting subset, report selection caveat | Azar, Marinescu & Steinbaum (2022, JHR) |
| **Condition on observability** | Run core analysis on full sample (non-wage outcomes), then repeat on wage subset to test if subsample differs | Standard robustness check |
| **Heckman selection correction** | When salary is a key outcome variable | Lovaglio (2025) |
| **Bounding** | Present results under best/worst case missingness assumptions | Rare but strongest defense |

**For non-salary fields:**
- **Skills**: Extracted via NLP from free text. "Missing" = skill not mentioned = treated as absence, not missing data. This is correct.
- **Seniority**: Our 27% "not applicable" rate is common. Standard practice: impute from title keywords (which we do), then validate against labeled subset. Report imputation rate and accuracy.
- **Education/experience**: Present in ~50% of postings (Carnevale et al.). Analyze conditional on presence; report missingness rate.

**Diagnostic tools:**
- `missingno` library: `msno.heatmap(df)` reveals whether salary missingness correlates with other fields (seniority, company size, industry). If correlations are strong, missingness is at least MAR. If missingness correlates with the outcome itself (salary level), it's MNAR.
- `miceforest`: MICE imputation via LightGBM. Creates multiple imputed datasets for proper variance estimation. Only valid under MAR assumption — do NOT use for salary without acknowledging this.
- **Heckman two-step** (manual via statsmodels): Probit selection equation predicting salary observability, then inverse Mills ratio correction in outcome equation. If IMR coefficient is significant, salary data is MNAR and naive imputation is biased.

**Sources**: Hazell & Taska (2023), NBER WP 31984; Lovaglio (2025), J. Forecasting; Azar et al. (2022), JHR; Carnevale et al. (2014), Georgetown CEW.

### 3. Selection bias diagnostics

**The specific selection mechanisms in our data:**

1. **Platform selection**: LinkedIn overrepresents BA+ jobs (posted online >80% of the time) vs. lower-education jobs (<50%). For SWE roles specifically, coverage is high, but our control occupations (civil eng, nursing) may have very different online posting rates.

2. **Algorithm selection**: LinkedIn's ranking algorithm optimizes for engagement, not representativeness. Promoted (paid) posts get 3-5x visibility. Our scraper captures whatever LinkedIn's algorithm surfaces for our search queries — this is not a random sample of all SWE postings.

3. **Scraper selection**: Our search queries, geographic scope, rate limiting, and pagination depth all create non-random gaps. The Kaggle dataset and our scraper have fundamentally different selection mechanisms — different queries, different time of scraping, different anti-bot handling.

4. **Employer selection**: Staffing companies obscure 30-40% of employer names. Large companies post more systematically than small ones. Companies may post "ghost" listings that inflate counts.

5. **Temporal selection (volatility bias)**: Daily scraping oversamples longer-lived postings (Foerderer 2023). A job open for 60 days is 60x more likely to appear in any given scrape than a 1-day posting.

**What to do beyond post-stratification:**

**Covariate balance checking** — The most directly applicable tool is Meta's `balance` package:
```python
from balance import Sample
sample = Sample.from_frame(scraped_df)
target = Sample.from_frame(bls_benchmark_df)
adjusted = sample.set_target(target).adjust()
# Reports ASMD (Absolute Standardized Mean Difference) per covariate
# Threshold: ASMD < 0.1 is acceptable (Stuart, Lee & Leacy 2013)
```

**What is NOT standard** (and why): Propensity score matching between online-posted vs. non-posted jobs is not done in this literature because there is no microdata on non-posted jobs to match against. The "selection" happens at the firm/posting level with no observable counterfactual.

**What published papers actually do:**
- Compare distributions across occupations, industries, geographies, education levels against OES/CPS (Hershbein & Kahn)
- Report deviation tables showing where online postings over- or under-represent
- Use establishment-level fixed effects to absorb unobserved heterogeneity in posting behavior (Acemoglu et al. 2022)
- Validate with natural experiments showing data responds to real labor shocks (Modestino et al. 2020 used troop withdrawals from Iraq/Afghanistan)

**Sources**: Meta balance package (arXiv 2307.06024); Stuart, Lee & Leacy (2013); Foerderer (2023); Acemoglu et al. (2022), JLE; Modestino et al. (2020), REStat.

### 4. Text classifier validation

**Our classifiers that need validation:**
1. `SWE_PATTERN` regex — identifies SWE-related postings from titles
2. `CONTROL_PATTERN` regex — identifies control occupation postings
3. `impute_seniority()` — rule-based seniority classifier from titles + description

**What the literature shows about accuracy:**

| Classifier type | Granularity | Best accuracy | Source |
|----------------|------------|--------------|--------|
| SOC coding from titles | 2-digit (10 labels) | 77-82% | Stra et al. (2025); 42M postings study (2023) |
| SOC coding from titles | 6-digit (800+ labels) | 45-57% | Same |
| SOC coding from descriptions | 2-digit | 79% | 42M postings study |
| Skill extraction | Token-level | F1 = 56-64% | SkillSpan (Zhang et al. 2022, NAACL) |

**Critical gap**: No published study has rigorously validated a seniority classifier. Most papers use keyword matching in titles (like we do) but treat it as face-valid without formal validation. This is actually an opportunity — if we validate ours, we're ahead of the literature.

**Common failure modes:**
- **Label leakage**: Our seniority classifier uses title keywords ("Senior", "Junior"). If we then analyze seniority distributions by title, we're measuring our own classifier's behavior, not the labor market. The classifier must be validated against an independent ground truth.
- **Temporal drift**: A regex trained on 2023-2024 title conventions may fail on 2026 titles if naming conventions changed (e.g., new titles like "AI Engineer" that don't match our SWE_PATTERN).
- **Subpopulation failure**: A classifier with 90% overall accuracy might have 50% accuracy on the specific subgroup you care about (e.g., "entry-level" postings with ambiguous titles).
- **Overfitting to canonical titles**: Training/testing on clean O\*NET titles produces inflated accuracy vs. real-world messy titles.

**What we need to do:**

Create a **gold-standard validation set**:
1. Randomly sample **300-500 postings** stratified by source (Kaggle vs. scraped), time period, and predicted seniority level
2. Have **2-3 annotators** independently label each posting for: (a) is this SWE? (b) seniority level (c) key skills
3. Compute **inter-rater reliability**: Cohen's kappa for 2 annotators (`sklearn.metrics.cohen_kappa_score`), Krippendorff's alpha for 3+ (`simpledorff` package)
4. Adjudicate disagreements to create final gold standard
5. Evaluate our classifiers against this gold standard, reporting **per-class precision/recall** (not just overall accuracy)

**Thresholds:**
- Inter-annotator agreement: kappa >= 0.67 tentatively acceptable, >= 0.80 good (Krippendorff's standard)
- Classifier accuracy: no hard threshold, but must be contextualized against human agreement baseline. If humans agree at 80%, a classifier at 75% is defensible.

**Annotation tools**: Label Studio (free, open-source, multi-annotator support) or Doccano (lightweight). Both support web-based annotation with adjudication workflows.

**Sources**: Stra et al. (2025), arXiv; Zhang et al. (2022), NAACL; Skill extraction survey (arXiv 2402.05617); 42M postings occupational models (PMC 10382938).

### 5. Robustness and sensitivity analysis

**What reviewers will attack:**

The most sensitive specification choices in cross-sectional job posting studies, ranked by impact:

1. **Occupation definition**: Our SWE_PATTERN regex is a researcher choice. Widening it (include "data analyst"?) or narrowing it (exclude "DevOps"?) changes sample composition. Run core results under 2-3 alternative definitions.

2. **Deduplication method**: Given that 50-80% of raw postings may be duplicates (Lightcast), the dedup method and threshold materially affect all counts. Run with strict dedup (exact match) and loose dedup (embedding similarity) and show results are robust.

3. **Seniority classification**: Our title-keyword approach is one of many possible choices. Show results hold under (a) our imputed labels, (b) LinkedIn's native labels where available, (c) a description-based classifier.

4. **Geographic and industry scope**: If our scraper over-represents certain metros or industries, results may not generalize. Restrict to top-10 metros and show results hold.

5. **Time period alignment**: March 2026 vs. full 2023-2024 vs. March-only 2023-2024. Show all three comparisons.

**Specification curve analysis:**

The `specification_curve` Python package (Turrell) automates this:
```python
import specification_curve as sc
sco = sc.SpecificationCurve(
    df,
    y_endog=['junior_share', 'log_junior_share'],
    x_exog='period',  # 2023-24 vs 2026
    controls=['industry', 'region', 'company_size'],
    always_include='seniority_source'
)
sco.fit()
sco.plot()  # Shows coefficient across all specifications
```

This directly addresses the "garden of forking paths" problem (Gelman & Loken 2013) — showing that your result holds across all defensible analytical choices, not just the one you picked.

**Placebo/falsification tests:**
- Run the same cross-period comparison on **control occupations** (civil eng, nursing). If they show the same "structural change" as SWE, your finding is confounded by macro trends, not AI-specific.
- Run on a **placebo time split** within the Kaggle data (first half vs. second half of 2023-2024). If you find a "break" there too, your method is detecting noise.
- Pre-trend analysis: if Kaggle data spans enough months, show that the SWE metrics were stable before the hypothesized break.

**What published papers do:**
- Modestino et al. (2020): Stratified by business cycle phase, firm size, sector, occupation type
- Acemoglu et al. (2022): Tested 3 different AI exposure indices, each identifying different occupations as AI-exposed
- Deming & Kahn (2018): Aggregated to firm-MSA-SOC-quarter level as alternative specification
- Standard: run on (a) full sample, (b) wage-posting subset only, (c) specific occupation groups, (d) excluding outliers

**Sources**: specification_curve (GitHub: aeturrell); Gelman & Loken (2013); Modestino et al. (2020), REStat; Acemoglu et al. (2022), JLE; Deming & Kahn (2018), JLE.

## Recommendations for our study

### Immediate actions (add to notebook)

1. **Representativeness check**: Pull JOLTS and OES data via `fredapi`/BLS API. Compare our scraped occupation and industry distributions against official benchmarks. Compute correlations and dissimilarity indices. Do this separately for each dataset (Kaggle and scraped).

2. **Missing data audit**: Run `missingno` visualizations on both datasets. Report missingness rates per field. For salary: compute `msno.heatmap()` to diagnose whether missingness correlates with seniority, industry, or company. Document that salary missingness is likely MNAR and should not be naively imputed.

3. **Gold-standard annotation**: Sample 300-500 postings (stratified by source and predicted seniority). Set up Label Studio. Have 2 annotators label for SWE/non-SWE, seniority, and key skills. Compute kappa. Evaluate our regex and imputation classifiers against this set. Report per-class precision/recall.

4. **Covariate balance check**: Use Meta's `balance` package to compute ASMD between our scraped data and a BLS benchmark, and between the Kaggle and scraped datasets. Report balance before and after any reweighting.

5. **Specification curve**: Install `specification_curve`. Run core findings (junior share change, skill prevalence change) across all defensible specifications (alternative SWE definitions, dedup thresholds, seniority classifiers, geographic subsets). Plot the curve.

### Methodology section additions

6. **Representativeness table**: Side-by-side occupation/industry shares for our data vs. OES/JOLTS, with correlation coefficients and dissimilarity indices. Standard in all Burning Glass papers.

7. **Missing data table**: Missingness rates per field per dataset. Explicit statement that salary is likely MNAR with citation to Hazell & Taska (2023).

8. **Classifier validation table**: Per-class precision/recall/F1 for SWE detection and seniority imputation against gold-standard annotations. Report inter-annotator agreement.

9. **Selection mechanisms paragraph**: Document the 5 specific selection mechanisms (platform, algorithm, scraper, employer, temporal). Explain which ones are held constant by LinkedIn-to-LinkedIn comparison and which are not.

10. **Robustness section**: Specification curve plot. Placebo tests on control occupations and placebo time splits. Subsample analyses by geography, industry, company size.

### Nice-to-have

11. **Heckman selection correction** for any analysis that uses salary as an outcome variable.
12. **Temporal drift test**: Run SWE_PATTERN and seniority classifier on a held-out 2022 sample (if obtainable) to test whether accuracy degrades over time.
13. **Inverse-probability-of-capture weighting** to correct for volatility bias in daily scraping (Foerderer 2023).

## Key sources

- **Hershbein & Kahn (2018)** — "Do Recessions Accelerate Routine-Biased Technological Change?" AER.
  - Relevance: Canonical validation methodology. Correlation benchmarks we must replicate.
  - URL: https://www.nber.org/papers/w22762

- **Hazell & Taska (2023)** — "Online Job Posts Contain Very Little Wage Information." NBER WP 31984.
  - Relevance: Proves salary missingness is MNAR. Only 17% of BG postings have wages.
  - URL: https://www.nber.org/papers/w31984

- **OECD (2024)** — "How well do online job postings match national sources?" LEED Paper 2024/01.
  - Relevance: Dissimilarity index methodology for international benchmarking.
  - URL: https://www.oecd.org/en/publications/how-well-do-online-job-postings-match-national-sources-in-large-english-speaking-countries_c17cae09-en.html

- **Stra et al. (2025)** — "Standard Occupation Classifier — An NLP Approach." arXiv.
  - Relevance: Benchmarks for SOC classification accuracy (59-77% depending on granularity).
  - URL: https://arxiv.org/abs/2511.23057

- **Zhang et al. (2022)** — "SkillSpan: Hard and Soft Skill Extraction." NAACL.
  - Relevance: Gold-standard annotation methodology for job posting NLP. 3 annotators, kappa 0.70-0.75.
  - URL: https://aclanthology.org/2022.naacl-main.366/

- **Modestino, Shoag & Ballance (2020)** — "Upskilling." REStat.
  - Relevance: Model for robustness testing — subsample stratification, natural experiment validation.
  - URL: https://direct.mit.edu/rest/article/102/4/793/96774

- **Acemoglu, Autor, Hazell & Restrepo (2022)** — "Artificial Intelligence and Jobs." JLE.
  - Relevance: Multi-index robustness testing, establishment-level fixed effects.
  - URL: https://www.journals.uchicago.edu/doi/abs/10.1086/718327

- **Deming & Kahn (2018)** — "Skill Requirements across Firms and Labor Markets." JLE.
  - Relevance: Alternative aggregation levels as robustness check.
  - URL: https://www.journals.uchicago.edu/doi/10.1086/694106

- **Gelman & Loken (2013)** — "The garden of forking paths."
  - Relevance: Theoretical basis for specification curve analysis.
  - URL: http://www.stat.columbia.edu/~gelman/research/unpublished/forking.pdf

- **Meta balance package** — Facebook Research, 2023.
  - Relevance: ASMD-based covariate balance checking for survey/scraped data.
  - URL: https://github.com/facebookresearch/balance

- **Skill extraction survey (2024)** — arXiv 2402.05617.
  - Relevance: Documents that none of 8 public skill extraction datasets report inter-annotator agreement.
  - URL: https://arxiv.org/abs/2402.05617

- **specification_curve** — Turrell, Python package.
  - Relevance: Automated specification curve analysis.
  - URL: https://github.com/aeturrell/specification_curve

- **Carnevale, Jayasundera & Repnikov (2014)** — "Understanding Online Job Ads Data." Georgetown CEW.
  - Relevance: BA+ jobs posted online >80% vs. <50% for lower-education. Parsing accuracy benchmarks.
  - URL: https://cew.georgetown.edu/wp-content/uploads/2014/11/OCLM.Tech_.Web_.pdf

- **Indeed Hiring Lab (2024)** — "Comparing Indeed Data with Public Employment Statistics."
  - Relevance: Occupational share correlations (r = 0.53-0.86 by country) and wage percentile accuracy.
  - URL: https://www.hiringlab.org/2024/09/20/comparing-indeed-data-with-public-employment-statistics/

## Open questions

1. **Annotation labor**: Creating a 300-500 posting gold standard with 2 annotators is real work. Should we prioritize this now or after we have more scraping months?

2. **BLS API access**: Do we need to register for a FRED API key? The free tier allows 20-year spans which is sufficient.

3. **Kaggle provenance**: The Kaggle dataset (arshkon/linkedin-job-postings) has no published methodology. Should we run the representativeness diagnostics on it as a "black box" dataset and simply report what we find, or try to reverse-engineer its collection method?

4. **Which specifications to pre-register**: If we commit to a specification curve, we should define the set of defensible specifications before looking at results. Should we draft this now?

5. **Control occupation coverage**: Our CONTROL_PATTERN (civil eng, nursing, etc.) may have very different LinkedIn posting rates than SWE. The DiD design (RQ4) requires that control occupations have adequate coverage in both datasets — we need to verify this.
