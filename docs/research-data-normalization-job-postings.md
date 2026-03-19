# Research: Data Normalization for Cross-Period Job Posting Comparison

Date: 2026-03-18
Context: Supports RQ1-RQ4. Our study compares Kaggle LinkedIn data (2023-2024, ~124K postings, third-party collection) with our own daily scraper output (March 2026, ~3-7K SWE postings/day). We need rigorous normalization to ensure apples-to-apples comparison.

## Question

How should we normalize and validate scraped job posting data for rigorous cross-period comparison? What do published studies do, what do reviewers expect, and what are the known pitfalls that sink papers using this type of data?

## Sub-questions explored

1. Schema harmonization across different job posting datasets
2. Sampling bias correction for scraped job postings
3. Deduplication across sources and time periods
4. Temporal normalization for datasets with very different time spans
5. Validity threats and reporting standards reviewers expect

## Findings

### 1. Schema harmonization

The standard approach is **retroactive classification**: apply a single taxonomy to both datasets using the same classifier, rather than relying on each dataset's native labels.

- **Lightcast's approach**: All historic and current postings are reclassified monthly using the latest classifier versions, ensuring taxonomy changes don't create artificial breaks. We should do the same — apply our SWE_PATTERN regex and seniority imputation logic to both datasets identically.
- **Job title normalization**: The Kaggle dataset uses LinkedIn's raw titles; our scraper may capture different formatting. Options range from simple regex (our current approach) to O\*NET SOC mapping via TF-IDF (97.5% precision per occupationdata.github.io) or graph-based approaches (JAMES, PSU/DSAA 2023).
- **Seniority labels**: The Kaggle dataset has LinkedIn's `formatted_experience_level`. Our scraper has `job_level`. These use different label sets. We already impute a 3-level scheme (junior/mid/senior) from title keywords — this should be the canonical seniority variable for cross-period comparison, applied uniformly to both datasets.
- **Company name standardization**: Use RapidFuzz `token_set_ratio` (threshold ~85) to group variants (e.g., "JPMC" / "JPMorgan" / "J.P. Morgan Chase"). Needed for any firm-level analysis.

**Sources**: Lightcast JPA Methodology (kb.lightcast.io); Turrell et al. (2019), NBER WP 25837; JAMES (PSU/DSAA 2023).

### 2. Sampling bias correction

Three distinct biases affect our data:

**a) Platform coverage bias**: LinkedIn overrepresents professional/tech/white-collar roles by ~11 percentage points vs. CPS data (Hershbein & Kahn 2018). This actually helps our study (more SWE coverage) but means cross-occupation comparisons (RQ4) require correction.

- **Mitigation**: Compare our occupation/industry distributions against BLS OES or JOLTS. Report the delta. Apply **occupation-level post-stratification weights** calibrated to official vacancy shares (Cammeraat & Squicciarini 2021, OECD).

**b) Volatility / left-truncation bias**: Daily scraping oversamples longer-lived postings — a job posted for 60 days is 60x more likely to appear in any single scrape than a job posted for 1 day. This biases toward hard-to-fill roles (Foerderer 2023, arXiv:2308.02231).

- **Mitigation**: Measure the correlation between posting duration and capture probability. Apply **inverse-probability-of-capture weighting** (analogous to survival analysis). At minimum, report the distribution of posting durations in each dataset.

**c) Selection into online posting**: Not all vacancies get posted online. Heckman-type selection models can estimate the probability of a vacancy being posted online by job type, then correct raw counts using the inverse Mills ratio (Lovaglio 2025, Journal of Forecasting). This reduced bias by ~50% against official benchmarks.

- **For our study**: The Heckman approach is heavy machinery. The pragmatic path is post-stratification reweighting against JOLTS + explicit acknowledgment of the selection issue.

**Sources**: Hershbein & Kahn (2018), AER; Cammeraat & Squicciarini (2021), OECD; Foerderer (2023), arXiv; Lovaglio (2025), J. Forecasting; ILO Working Paper 68 (2022).

### 3. Deduplication

This is critical — Lightcast reports deduplicating up to **80% of raw scraped postings**. Without dedup, volume counts are meaningless.

**Within-dataset dedup**:
- **Exact**: Drop duplicates on normalized `(title, company, location)` within a rolling window. Industry standard is **60 days** (Lightcast) or **30 days** (Turrell et al. 2019).
- **Near-duplicate**: Combine text similarity + skill overlap + embeddings. Best reported result: F1=0.94 using this multi-signal approach (Abdelaal et al. 2024, arXiv:2406.06257). Tiered cosine similarity thresholds: **0.8 for titles, 0.7 for descriptions**.

**Cross-dataset dedup**:
- Use SemHash (Model2Vec embeddings + ANN search) for fast cross-dataset near-duplicate detection. Processes 1.8M records in ~83 seconds on CPU.
- This catches jobs that were posted in 2024 and reposted in 2026.

**LinkedIn-specific issue**: "Reposted X days ago" means the employer refreshed the listing. A job showing "Reposted 3 days ago" could be months old. Our scraper's `_seen_job_ids.json` handles within-scrape dedup, but we need to verify the Kaggle dataset's dedup status.

**Boilerplate removal**: Company descriptions repeat across all of a company's postings and inflate similarity scores. Strip boilerplate before any text-based comparison or embedding.

**Sources**: Abdelaal et al. (2024), arXiv:2406.06257; Lightcast dedup documentation; SemHash (GitHub: MinishLab/semhash).

### 4. Temporal normalization

**The core problem**: We're comparing a multi-month dataset (Kaggle, 2023-2024) against a 2-week daily scrape (March 2026). Three issues:

**a) Seasonality**: January-March is peak hiring season (budget cycles reset). Our March 2026 scrape captures a seasonal high. If the Kaggle data averages across all months, we're comparing a seasonal peak to an annual average.

- **Mitigation**: Determine the exact month distribution of the Kaggle dataset. Restrict comparison to same-month data (March 2023/2024 vs. March 2026) if possible. If not, apply BLS seasonal adjustment factors or report the confound explicitly.

**b) Volume normalization**: Raw counts are not comparable across datasets of different sizes. Use **shares** (e.g., junior postings as % of total SWE postings) rather than absolute counts for all cross-period comparisons.

- **Mitigation**: All metrics should be expressed as proportions, rates, or distributions — never raw counts. This is already partially done in the notebook but should be systematic.

**c) Per-period reweighting**: Rather than assuming static occupation weights, recalculate adjustment factors for each time window independently (Turrell et al. 2019). The job market's composition changed between 2023-2024 and 2026.

**Sources**: Turrell et al. (2019), NBER WP 25837; Federal Reserve Bank of Chicago (2018); Brookings seasonal adjustment methods.

### 5. Validity threats and reporting standards

**Top 5 reviewer objections to anticipate:**

**1. "Your two datasets are not comparable."**
Different scraping methodologies, unknown Kaggle provenance, different deduplication.
*Defense*: Document both methodologies exhaustively. Apply uniform post-hoc cleaning. Run sensitivity analyses with different cleaning thresholds.

**2. "You're confounding seasonality with structural change."**
March 2026 vs. multi-month 2023-2024.
*Defense*: Same-month comparison or seasonal adjustment. Report calendar month of all data prominently.

**3. "Ghost jobs contaminate your counts."**
18-27% of US online job listings are estimated to be ghost jobs (CRS Report IF12977, April 2025). The rate increased from ~12.5% in 2023 to ~20% in 2025. 55%+ of "entry-level" tech postings demand 3+ years experience.
*Defense*: Acknowledge explicitly. Triangulate with JOLTS/BLS hiring data. Filter for postings with salary ranges (ghost jobs less likely to include salary). Note the ghost job rate differential.

**4. "LinkedIn is not the labor market."**
Platform bias toward white-collar, tech, English-speaking, college-educated.
*Defense*: We study within-platform change (LinkedIn-to-LinkedIn), holding platform bias roughly constant. The threat is if LinkedIn's policies/algorithms changed between periods — investigate and report.

**5. "You cannot establish causation from two cross-sections."**
*Defense*: Frame as descriptive evidence "consistent with" structural change, not proof. Use external data (layoff trackers, BLS, ADP) as triangulation. The Kaggle dataset alone spans 2023-2024, which helps establish a pre-trend.

**Reporting standards (from Brown et al. 2025, Big Data & Society)**:
Papers using scraped data must report: exact scraping dates/times, search queries and filters, rate limiting encountered, fields collected vs. available, missing data rates per field, deduplication method and rate, terms of service considerations, reproducibility notes.

**Key validation**: de Pedraza et al. (2019, IZA J. Labor Economics) showed scraped and official vacancy data have similar time-series properties at the aggregate level — this is the best defense for using scraped data at all.

**Sources**: CRS Report IF12977 (2025); Brown et al. (2025), Big Data & Society; de Pedraza et al. (2019), IZA; OECD (2024); Hershbein & Kahn (2018).

## Recommendations for our study

### Immediate actions (add to notebook)

1. **Uniform schema**: Apply the same SWE_PATTERN, seniority imputation, and field normalization to both datasets. Do not rely on native labels for cross-period comparison.

2. **Deduplication audit**: Check the Kaggle dataset's dedup status. Apply our dedup logic (title + company + location within 60-day window) to both datasets. Report raw vs. deduplicated counts.

3. **Proportions, not counts**: Convert all cross-period metrics to shares/rates. Junior share = junior postings / total SWE postings. Skill prevalence = postings mentioning skill / total SWE postings at that seniority level.

4. **Same-month comparison**: Filter Kaggle data to March 2023 and March 2024 for the primary cross-period comparison. Use the full Kaggle dataset only for within-period trend analysis.

5. **Distribution comparison**: For each key variable (seniority, skills, description length, salary), compare full distributions (not just means). Use KS tests or chi-squared tests.

6. **Missing data audit**: Report missing data rates per field per dataset. Salary data is ~70% missing in Kaggle — flag this.

### Methodology section additions

7. **Scraping methodology disclosure**: Document exact scraping dates, search queries, filters, geographic scope, dedup method, and dedup rate for both datasets. For the Kaggle dataset, note what is known and unknown about its provenance.

8. **Representativeness validation**: Compare our occupation distributions against JOLTS or OES. Report correlations. Do this separately for each dataset.

9. **Ghost job acknowledgment**: Cite the CRS estimate (18-27%), note the differential across periods, and explain why this does not invalidate our findings (we study content changes within postings, not just volume).

10. **Seasonality statement**: Report the calendar month distribution of both datasets and either restrict to same-month comparison or apply seasonal adjustment.

### Nice-to-have (if time permits)

11. **Occupation-level post-stratification weights** calibrated to JOLTS vacancy shares.
12. **Posting duration analysis** to test for volatility bias in the scrape.
13. **Cross-dataset near-duplicate detection** using SemHash to find jobs that persisted from 2024 to 2026.

## Key sources

- **Hershbein & Kahn (2018)** — "Do Recessions Accelerate Routine-Biased Technological Change?" AER.
  - Relevance: Canonical BGT validation paper. Correlation-based validation we should replicate.
  - URL: https://www.nber.org/papers/w22762

- **Cammeraat & Squicciarini (2021)** — "Burning Glass Technologies' data use in policy-relevant analysis." OECD.
  - Relevance: Occupation-level reweighting methodology for correcting representativeness.
  - URL: https://www.oecd.org/en/publications/burning-glass-technologies-data-use-in-policy-relevant-analysis_cd75c3e7-en.html

- **OECD (2024)** — "How well do online job postings match national sources in large English speaking countries?"
  - Relevance: Benchmark framework for validating scraped data against official sources.
  - URL: https://www.oecd.org/en/publications/how-well-do-online-job-postings-match-national-sources-in-large-english-speaking-countries_c17cae09-en.html

- **Foerderer (2023)** — "Should We Trust Web Scraped Data?" arXiv.
  - Relevance: Framework for diagnosing volatility bias in daily scraping.
  - URL: https://arxiv.org/pdf/2308.02231

- **Turrell et al. (2019)** — "Transforming Naturally Occurring Text Data Into Economic Statistics." NBER.
  - Relevance: Per-period reweighting approach and NLP-to-SOC pipeline.
  - URL: https://www.nber.org/papers/w25837

- **ILO Working Paper 68 (2022)** — "Methodological issues related to the use of online labour market data."
  - Relevance: Best single survey of all known biases and corrections. Cite in methods section.
  - URL: https://webapps.ilo.org/static/english/intserv/working-papers/wp068/index.html

- **Abdelaal et al. (2024)** — "Combining Embeddings and Domain Knowledge for Job Posting Duplicate Detection." arXiv.
  - Relevance: State-of-the-art dedup method (F1=0.94). Multi-signal approach.
  - URL: https://arxiv.org/html/2406.06257v1

- **Lovaglio (2025)** — "Forecasting New Employment Using Nonrepresentative Online Job Advertisements." J. Forecasting.
  - Relevance: Heckman selection model for bias correction (reduced bias ~50%).
  - URL: https://onlinelibrary.wiley.com/doi/10.1002/for.70090

- **Brown et al. (2025)** — "Web scraping for research: Legal, ethical, institutional, and scientific considerations." Big Data & Society.
  - Relevance: Reporting standards checklist for scraped data studies.
  - URL: https://journals.sagepub.com/doi/10.1177/20539517251381686

- **CRS Report IF12977 (2025)** — "'Ghost' Job Postings." Congressional Research Service.
  - Relevance: Official estimate of ghost job prevalence (18-27%).
  - URL: https://www.congress.gov/crs-product/IF12977

- **de Pedraza et al. (2019)** — "Survey vs Scraped Data." IZA J. Labor Economics.
  - Relevance: Validates that scraped and official data show similar time-series properties.
  - URL: https://sciendo.com/article/10.2478/izajole-2019-0004

- **Marinescu & Wolthoff (2020)** — "Opening the Black Box of the Matching Function." J. Labor Economics.
  - Relevance: Dual-validation approach (web source + official source).
  - URL: https://www.journals.uchicago.edu/doi/10.1086/705903

- **Acemoglu, Autor, Hazell & Restrepo (2022)** — "Artificial Intelligence and Jobs." J. Labor Economics.
  - Relevance: Multi-index triangulation for skill analysis; establishment-level fixed effects.
  - URL: https://www.journals.uchicago.edu/doi/abs/10.1086/718327

## Open questions

1. **Kaggle dataset provenance**: We don't know the exact scraping methodology, deduplication status, or completeness of the Kaggle LinkedIn dataset (arshkon/linkedin-job-postings). Should we contact the dataset author or treat the unknown provenance as a stated limitation?

2. **Ghost job detection**: Is there a practical way to flag likely ghost jobs in our data (e.g., postings open >90 days, no salary, inflated requirements for stated level)?

3. **LinkedIn algorithm changes**: Did LinkedIn change its job posting display, ranking, or categorization between 2023-2024 and March 2026? This could create artificial differences.

4. **Expanding the 2026 scrape window**: March is a single month. Do we need more months of scraping before the cross-period comparison is defensible? The seasonality concern is reduced by same-month comparison but the sample size concern remains.
