# Hypothesis Testing & Statistical Verification Plan

Date: 2026-03-19
Status: Draft — ready for review before implementation

This document covers Stage 11 (formal hypothesis testing for RQ1-RQ7) and Stage 12 (statistical verification). It runs after preprocessing (`plan-preprocessing.md`) and exploration/validation (`plan-exploration.md`) are complete.

**Research questions:** RQ1 (junior roles disappearing/redefined?), RQ2 (skill migration?), RQ3 (structural break?), RQ4 (SWE-specific?), RQ5 (training implications), RQ6 (senior archetype shift?), RQ7 (historical comparison). See `docs/research-design-h1-h3.md` for full specifications.

**Design constraint:** We have two cross-sections (April 2024, March 2026), not a panel. All analyses are framed as "consistent with" structural change, not as causal proof.

---

## Stage 11: Formal hypothesis testing

This is the core analysis that produces the paper's findings. Each RQ gets a primary test (the simplest credible method that answers the question), a stronger test (ML/NLP-powered, higher power or richer signal), and robustness checks. Every test is paired with the specific alternative finding that would falsify our hypothesis.

**Design constraint:** We have two cross-sections (April 2024, March 2026), not a panel. We cannot track individual postings or companies over time. All analyses are cross-sectional comparisons of distributions, proportions, and text features across periods. We frame findings as "consistent with" structural change, not as causal proof.

---

### 11a. RQ1: Are junior SWE roles disappearing or being redefined?

**What we need to show:** Either (a) the share of junior SWE postings declined between 2024 and 2026, or (b) the content of junior postings in 2026 resembles senior postings from 2024, or (c) both.

**Null hypothesis:** The junior share and the content of junior postings are statistically indistinguishable across the two periods, after controlling for composition.

#### Test 1: Seniority distribution shift (primary)

The simplest credible test. Compare the proportion of entry-level / associate / mid-senior SWE postings between April 2024 and March 2026.

```
H₀: P(junior | SWE, 2024) = P(junior | SWE, 2026)
H₁: P(junior | SWE, 2024) ≠ P(junior | SWE, 2026)
```

**Method:** Chi-squared test of homogeneity on the 3×2 contingency table (3 seniority levels × 2 periods). Report Cramér's V for effect size.

**Implementation:**
```python
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df['period'], df['seniority_3level'])
chi2, p, dof, expected = chi2_contingency(contingency)
cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
```

**Sufficient evidence:** p < 0.05 with Cramér's V > 0.05 (small-but-meaningful effect). Report the actual shift in junior share (percentage points) with bootstrap 95% CI.

**What would falsify:** If the junior share is unchanged or increased, the "disappearing" narrative is wrong. If the distribution shifted but only because a different mix of companies was captured, it's a composition artifact (checked in 9g/9h).

#### Test 2: Content convergence — the redefinition test (stronger)

This separates title inflation (cosmetic relabeling) from genuine content change (what the job actually requires). A 2026 "Junior SWE" posting that reads like a 2024 "Mid-Senior SWE" posting is a redefined role, not a disappeared one.

**Method:** Train a logistic regression seniority classifier on 2024 JobBERT-v2 embeddings. Apply it to 2026 postings. Measure the "redefinition rate": the fraction of 2026 entry-level postings that the 2024-trained model predicts as mid-senior.

```python
from sklearn.linear_model import LogisticRegression

# Train on 2024 data (where seniority labels are known)
clf = LogisticRegression(max_iter=1000)
clf.fit(embeddings_2024, seniority_2024)  # labels: junior / mid / senior

# Predict on 2026 entry-level postings
junior_2026 = df_2026[df_2026['seniority'] == 'entry level']
preds = clf.predict(embeddings_junior_2026)
redefinition_rate = (preds != 'junior').mean()
```

**Sufficient evidence:** If the redefinition rate for 2026 junior postings is significantly higher than the "misclassification" rate on 2024 junior postings (the baseline error rate), the content has shifted. Compare with a permutation test: shuffle the period labels 1000 times and recompute. The observed redefinition rate should exceed 95% of permuted rates.

**Alternative method (model-free):** Compute the cosine similarity between each 2026 junior posting's embedding and the 2024 junior centroid vs. the 2024 senior centroid. If more 2026 junior postings are closer to the senior centroid than to the junior centroid (compared to 2024 junior postings), that's convergence. Test with a two-sample t-test on the similarity ratio.

#### Test 3: Scope inflation metrics

**Methods (run all, report as a table):**

| Metric | Test | Interpretation |
|---|---|---|
| Description length (requirements section only) | Mann-Whitney U | Longer requirements = more demanded of juniors |
| Distinct skill count per posting | Mann-Whitney U | More skills = broader scope |
| Years of experience required (median) | Mann-Whitney U | Higher YoE = inflated requirements |
| Skill Breadth Index (ESCO-mapped distinct skills) | Mann-Whitney U | Taxonomy-grounded scope measure |
| Senior keyword infiltration rate | Proportion test (z-test) | "system design", "architecture" appearing in junior postings |

Each test is run on junior postings only, comparing 2024 vs. 2026. Use Benjamini-Hochberg FDR correction across the battery. Report effect sizes (Cohen's d or rank-biserial correlation) alongside p-values.

#### Test 4: Controlled comparison (OLS regression)

Control for composition to isolate within-composition change:

```
SkillBreadth_i = β₀ + β₁(Period2026) + β₂(CompanySize) + β₃(Industry)
                + β₄(Metro) + β₅(IsRemote) + ε_i
```

Run on junior SWE postings only. β₁ is the period effect after controlling for composition. Use HC3 robust standard errors. Cluster by company if enough firms appear in both periods.

**Sufficient evidence for RQ1 overall:** At least 2 of the 4 tests pointing in the same direction (seniority shift + content convergence, or seniority shift + scope inflation, etc.). A single test alone is not conclusive given our data limitations.

---

### 11b. RQ2: Which competencies migrated from senior to junior postings, and in what order?

**What we need to show:** Specific skills that were predominantly senior-associated in 2024 appear at significantly higher rates in junior postings by 2026. The chronological ordering of this migration is secondary (limited by having only 2 time points) but we can establish which skills have migrated furthest.

**Null hypothesis:** The skill profile of junior postings is unchanged between periods. No individual skill shows a statistically significant increase in junior prevalence.

#### Test 1: Skill prevalence shift (primary)

For each ESCO-mapped skill (from 10e), compute its prevalence in junior postings in each period. Test for significant changes.

```
H₀: P(skill_k | junior, 2024) = P(skill_k | junior, 2026)   for each skill k
H₁: P(skill_k | junior, 2024) ≠ P(skill_k | junior, 2026)   for at least some k
```

**Method:** Two-proportion z-test for each skill. Apply Benjamini-Hochberg FDR at q = 0.05. Report adjusted p-values and the absolute prevalence change (Δ percentage points).

**Visualization:** Skill migration heatmap — rows = skills (sorted by Δ), columns = (junior 2024, senior 2024, junior 2026, senior 2026). Color = prevalence. The visual pattern of skills "sliding" from senior-only to junior+senior is the core RQ2 figure.

#### Test 2: Fightin' Words for junior vocabulary shift

Run Fightin' Words (log-odds-ratio with Dirichlet prior) on:
- Junior 2024 vs. Junior 2026 (what changed in junior postings?)
- Junior 2026 vs. Senior 2024 (do 2026 juniors sound like 2024 seniors?)

Words with high positive log-odds in the second comparison AND high positive log-odds in the first comparison are "migrated" terms: they're newly associated with junior roles and they sound like what senior roles used to require.

**Implementation:**
```python
from fightin_words import FWExtractor
fw = FWExtractor(ngram_range=(1, 3), min_df=5)
results = fw.fit_transform(junior_2026_texts, senior_2024_texts)
# Sort by z-score; positive = overrepresented in junior 2026 vs senior 2024
```

#### Test 3: Skill co-occurrence network shift

Build a skill co-occurrence graph for junior postings in each period. Nodes = ESCO skills, edges = co-occurrence within the same posting, edge weights = PMI (pointwise mutual information).

**What to measure:**
- New edges in 2026 that didn't exist in 2024 (new skill combinations emerging in junior roles)
- Skills that gained centrality (degree, betweenness) — these are skills that became "hub" requirements
- Community structure changes — did junior skill clusters reorganize?

**Implementation:**
```python
import networkx as nx
from sklearn.metrics import mutual_info_score

G_2024 = build_cooccurrence_graph(junior_skills_2024, min_pmi=1.0)
G_2026 = build_cooccurrence_graph(junior_skills_2026, min_pmi=1.0)

new_edges = set(G_2026.edges()) - set(G_2024.edges())
centrality_change = {skill: nx.betweenness_centrality(G_2026).get(skill, 0)
                            - nx.betweenness_centrality(G_2024).get(skill, 0)
                     for skill in all_skills}
```

#### Test 4: Embedding trajectory analysis

For each skill term (e.g., "system design"), compute its average context embedding when it appears in junior postings vs. senior postings, in each period.

**Migration metric:** For skill k, compute:
```
convergence_k = cos_sim(junior_context_2026_k, senior_context_2024_k)
              - cos_sim(junior_context_2024_k, senior_context_2024_k)
```

If convergence_k > 0, the way junior postings talk about skill k in 2026 is more similar to how senior postings talked about it in 2024. This captures whether skills are migrating in meaning (ownership vs. exposure), not just frequency.

**Sufficient evidence for RQ2:** A set of 5-15 skills showing statistically significant prevalence increases in junior postings (FDR-corrected), with directional confirmation from Fightin' Words and co-occurrence network analysis. The task migration map (which skills, ranked by magnitude of shift) is the deliverable.

---

### 11c. RQ3: Did the junior SWE market experience a structural break?

**What we need to show:** The cross-sectional differences between 2024 and 2026 are larger than what gradual trend extrapolation would predict. Ideally: evidence of a discrete level shift rather than smooth drift.

**Constraint:** With only 2 time points, we cannot run time-series methods (Bai-Perron, ITS) on our own data. These require a monthly time series. Our options:

1. Use external time-series data (Revelio Labs, JOLTS) for breakpoint detection, and show that our cross-sectional findings are consistent with the identified break.
2. Acquire the 1.3M Kaggle dataset — if it spans multiple months of 2024, we get within-2024 variation.
3. Frame our contribution as documenting the magnitude of change between 2024 and 2026, with the break-detection as supplementary evidence from external data.

#### Test 1: External time series analysis (Revelio + JOLTS)

We already have Revelio Labs data (SOC 15 hiring, openings, salary, employment, 2021-2026) and JOLTS data. Run structural break detection on these external series.

**Method:** Bai-Perron endogenous breakpoint detection on the Revelio SOC-15 monthly job openings series.

```python
import ruptures as rpt

# Revelio monthly SWE job openings, 2021-2026
signal = revelio_soc15_openings.values
algo = rpt.Pelt(model="rbf", min_size=3).fit(signal)
breakpoints = algo.predict(pen=10)
# Returns indices of detected break dates
```

**Workflow:**
1. UDmax test: any breaks at all in the Revelio SWE openings series?
2. If yes, estimate break date(s) with confidence intervals
3. Run the same on control occupation series (nursing, civil engineering). If they show the same break, it's macro, not SWE-specific.

**Confirmatory:** Chow test at hypothesized break date (December 2025, when production coding agents deployed):
```python
from statsmodels.stats.diagnostic import breaks_cusumolsresid
# Or manual F-test splitting series at Dec 2025
```

#### Test 2: Magnitude-of-change test

Even without a time series, we can test whether the observed 2024-2026 difference is larger than what random variation would produce.

**Method:** Permutation test. Pool all SWE postings from both periods. Randomly assign them to "2024" and "2026" groups (maintaining the original group sizes). Recompute the seniority shift metric (junior share change) 10,000 times. The observed shift should exceed 95% of permuted shifts.

```python
observed_shift = junior_share_2026 - junior_share_2024
permuted_shifts = []
for _ in range(10000):
    shuffled = np.random.permutation(all_seniority_labels)
    perm_2024 = shuffled[:n_2024]
    perm_2026 = shuffled[n_2024:]
    permuted_shifts.append(perm_2026_junior_share - perm_2024_junior_share)
p_value = (np.abs(permuted_shifts) >= np.abs(observed_shift)).mean()
```

This doesn't prove a break per se, but it proves the difference is not noise.

#### Test 3: Multivariate change detection (BOCPD on external data)

Run Bayesian Online Change Point Detection simultaneously on multiple Revelio/JOLTS series: SWE openings + hiring rate + salary trend + average skill count. Multivariate detection has higher power because it pools coordinated signals.

```python
from bayesian_changepoint_detection import online_changepoint_detection
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.likelihoods import MultivariateGaussian

# Stack multiple time series into a matrix
features = np.column_stack([swe_openings, hiring_rate, avg_skill_count])
R, maxes = online_changepoint_detection(features, constant_hazard(250),
                                         MultivariateGaussian())
```

**Sufficient evidence for RQ3:** A break detected in external time series (Revelio/JOLTS) near the hypothesized date (late 2025), combined with a permutation test showing our cross-sectional difference exceeds random variation. If no external break is detected, we frame RQ3 as inconclusive and focus on the magnitude of cross-sectional change.

---

### 11d. RQ4: Is this shift specific to SWE or part of a broader labor market trend?

**What we need to show:** The patterns observed in SWE postings (seniority shift, skill migration, scope inflation) do NOT appear in control occupations, or appear at significantly smaller magnitude.

**Null hypothesis:** The change in junior share (or skill breadth, or description length) is the same for SWE and control occupations.

#### Test 1: Difference-in-Differences (primary)

The workhorse causal inference design. Compare the change in junior share between SWE (treatment) and non-AI-exposed occupations (control) across the two periods.

```
Y_i = α + β₁(SWE) + β₂(Post2026) + β₃(SWE × Post2026) + γX + ε
```

β₃ is the treatment effect: the SWE-specific change beyond what control occupations experienced.

**Implementation:**
```python
import statsmodels.formula.api as smf

model = smf.ols(
    'junior_indicator ~ is_swe * is_post2026 + company_size + is_remote + C(industry)',
    data=df_swe_and_control
).fit(cov_type='HC3')
# β₃ = model.params['is_swe:is_post2026']
```

**Run on multiple outcomes:** junior share, skill breadth index, description length, AI-keyword prevalence. Each DiD produces a separate β₃ estimate. Use FDR correction across outcomes.

**Critical assumption:** Parallel pre-trends. With only 2 time points, we cannot directly test this. Mitigation:
- Use external data (Revelio) to show SWE and control occupations were on parallel trajectories before 2024
- Report sensitivity of β₃ to different control occupation sets

**Control occupation selection:** Use Felten et al. (2023) AI exposure scores. Select occupations in the bottom quartile of AI exposure: civil engineering (SOC 17-2051), mechanical engineering (17-2141), registered nursing (29-1141), accounting (13-2011). These are our Tier 3 scraper queries. Verify adequate sample sizes in both periods before running.

#### Test 2: Synthetic control (stronger)

Instead of hand-picking controls, construct a weighted combination of all non-SWE occupations that best matches pre-treatment SWE trends (using Revelio data where we have monthly resolution).

```python
from SyntheticControlMethods import Synth

# Y = junior share, monthly, by occupation
synth = Synth(df, outcome='junior_share', unit='occupation',
              time='month', treatment='SWE', treatment_period='2025-12')
synth.fit()
synth.plot(['original', 'pointwise', 'cumulative'])
```

The gap between the actual SWE junior share and the synthetic control's predicted junior share IS the treatment effect.

**Robustness:** Run placebo synthetic controls for each donor occupation (in-space placebo). If many placebos show gaps as large as SWE, the finding is not significant. Compute a pseudo p-value: (rank of SWE gap among all gaps) / (number of donor occupations).

**Data requirement:** Monthly time series per occupation. This requires the Revelio data or the 1.3M Kaggle dataset (if it has monthly resolution). Cannot be done with our 2-snapshot data alone.

#### Test 3: Cross-occupation embedding analysis

Using the full embedding space from 10b, measure whether SWE postings drifted more than control postings between periods.

```python
# Centroid drift per occupation
for occupation in ['SWE', 'civil_eng', 'nursing', 'mech_eng']:
    centroid_2024 = embeddings[occ == occupation & period == 2024].mean(axis=0)
    centroid_2026 = embeddings[occ == occupation & period == 2026].mean(axis=0)
    drift = 1 - cosine_similarity([centroid_2024], [centroid_2026])[0][0]
    print(f"{occupation}: drift = {drift:.4f}")
```

If SWE drift >> control drift, the change is SWE-specific. Test with a bootstrap: resample postings within each occupation-period, recompute drift, build confidence intervals. If the SWE drift CI doesn't overlap with control drift CIs, the difference is significant.

**Sufficient evidence for RQ4:** A positive, significant β₃ in the DiD regression, confirmed by directional consistency in the synthetic control (if feasible) and the embedding drift comparison. If β₃ is null, the shift is macro (affects all occupations equally) and we report that finding instead.

---

### 11e. RQ6: Are senior SWE roles shedding management and gaining AI-orchestration?

**What we need to show:** Senior SWE postings in 2026 mention management/people-leadership skills less frequently and AI/orchestration skills more frequently than in 2024.

**Null hypothesis:** The management keyword frequency and AI-orchestration keyword frequency in senior SWE postings are unchanged between periods.

#### Test 1: Keyword category prevalence shift (primary)

Define two keyword categories grounded in the research design:

**Management category:** mentorship, coaching, hiring, team lead, team leadership, performance review, people management, direct reports, staff development, career development, 1:1, one-on-one

**AI-orchestration category:** AI agent, LLM, large language model, prompt engineering, RAG, retrieval augmented, model evaluation, AI integration, copilot, AI orchestration, AI-assisted, agent framework, vector database, agentic

For each category, compute prevalence (fraction of senior SWE postings mentioning at least one keyword) in each period. Test with two-proportion z-test.

**Archetype Shift Index:** Compute the ratio (AI-orchestration prevalence) / (management prevalence) for each period. This single number captures the directional shift. Test whether it increased with a bootstrap CI.

#### Test 2: Fightin' Words on senior postings

Run Fightin' Words comparing senior 2024 vs. senior 2026 descriptions. The top terms distinguish the two periods without requiring predefined keyword lists. This is a check on our keyword categories: if our chosen keywords rank highly in the Fightin' Words output, our categories are well-constructed. If other terms rank higher, we may be missing important dimensions of change.

#### Test 3: BERTopic on senior postings only

Run BERTopic separately on senior SWE postings (both periods combined). Examine whether:
- Management-themed topics are more prevalent in 2024 than 2026
- AI/technical-themed topics are more prevalent in 2026 than 2024
- New topics exist in 2026 that don't appear in 2024 (emerging senior archetype)

Use `topics_over_time()` with 2 bins (2024, 2026) to quantify topic prevalence shift.

#### Test 4: Controlled regression

```
MgmtKeywordCount_i = β₀ + β₁(Post2026) + β₂(CompanySize) + β₃(Industry) + ε
AIKeywordCount_i   = β₀ + β₁(Post2026) + β₂(CompanySize) + β₃(Industry) + ε
```

Run on senior SWE postings only. β₁ should be negative for management and positive for AI-orchestration. HC3 standard errors.

**Sufficient evidence for RQ6:** Significant decline in management keyword prevalence AND significant increase in AI-orchestration keyword prevalence, confirmed by Fightin' Words and BERTopic showing the same directional shift. The Archetype Shift Index increasing between periods, with a bootstrap CI excluding zero change.

---

### 11f. RQ5: Training implications & RQ7: Historical platform comparison

These are synthesis RQs, not hypothesis tests. They draw on the empirical outputs of RQ1-4 and RQ6.

#### RQ5: Training implications

**Method:** Structured qualitative synthesis.

1. From RQ2's task migration map, identify the top 5-10 skills that migrated to junior roles. These are the skills that training programs must front-load.
2. From RQ1's scope inflation analysis, quantify how much more is now expected of entry-level candidates. This frames the training gap.
3. From RQ6's archetype shift, identify the new senior competencies (AI orchestration) that the training pipeline must eventually develop.
4. Cross-validate against documented cross-profession parallels:
   - Radiology: when AI diagnostic tools deployed, how did residency programs adapt?
   - Accounting: when automated bookkeeping arrived, how did entry requirements change?
   - Aviation: how did autopilot change the pilot training pipeline?
5. Derive prescriptive recommendations, each traceable to a specific empirical finding.

**Output:** The "AI Supervision Residency" framework (Appendix B in research design), grounded in empirical evidence rather than speculation.

#### RQ7: Historical platform comparison

**Method:** Descriptive comparison using O\*NET occupation definitions across time.

1. Pull O\*NET "detailed work activities" and "technology skills" for SOC 15-1252 (Software Developers) across available O\*NET versions (2010, 2015, 2019, 2024).
2. For each version, identify the modal "senior SWE" skill profile.
3. Construct a decade-by-decade summary of the senior SWE archetype:
   - **Mainframe era (1960s-80s):** Hardware optimization, batch processing
   - **PC/C era (1980s-90s):** Systems programming, memory management
   - **Web/Java era (2000s):** Architecture, design patterns, team management
   - **Mobile/cloud era (2010s):** Distributed systems, cross-functional coordination
   - **AI era (2025+):** AI orchestration, system design, less people management
4. Quantify the magnitude of each transition where data exists (O\*NET task change rates). Is the AI-era shift larger or smaller than prior transitions?

**This is contextualization, not causal identification.** It places our RQ1-RQ6 findings in the longer arc of computing history.

---

### 11g. Robustness & multiple testing protocol

This section applies to ALL RQs. Without it, a reviewer can dismiss any individual finding as fragile or cherry-picked.

#### Specification curve analysis

Run every primary test under all defensible specifications (defined in Stage 9k). The specification space:

| Dimension | Variants | Count |
|---|---|---|
| SWE definition | Narrow, standard, broad | 3 |
| Seniority classifier | Our imputer, native labels, description-only | 3 |
| Dedup threshold | Strict (exact), standard (0.70), loose (0.50) | 3 |
| Sample scope | Full, LinkedIn-only, excl. aggregators, top-10 metros | 4 |
| Company capping | None, cap at 10 per company, cap at median | 3 |

Total: 3 × 3 × 3 × 4 × 3 = 324 specifications per test.

```python
import specification_curve as sc
sco = sc.SpecificationCurve(
    df, y_endog='junior_share',
    x_exog='is_post2026',
    controls=[['company_size'], ['industry'], ['is_remote'], ['metro']],
)
sco.fit()
sco.plot()
```

**Rule:** A finding is "robust" if it is significant (p < 0.05) in > 80% of specifications. If 50-80%, it is "suggestive." If < 50%, it is "fragile" and not reported as a finding.

#### Multiple testing corrections

| Scope | Method | Rationale |
|---|---|---|
| Within a single RQ (e.g., testing 15 skills in RQ2) | Benjamini-Hochberg FDR at q = 0.05 | Controls false discovery rate |
| Across RQs (e.g., RQ1 + RQ2 + RQ4 + RQ6 primary tests) | Holm-Bonferroni | More conservative for the study's headline findings |
| Specification curve | Report the fraction significant, not individual p-values | The curve itself accounts for multiplicity |

#### Bootstrap confidence intervals

For every key estimate (junior share change, skill prevalence change, Archetype Shift Index, DiD coefficient), report bootstrap 95% CIs alongside the point estimate. Use 2000 bootstrap resamples.

```python
from scipy.stats import bootstrap
result = bootstrap((junior_share_data,), np.mean, n_resamples=2000,
                   confidence_level=0.95, method='BCa')
```

BCa (bias-corrected and accelerated) intervals are preferred over percentile intervals for skewed distributions.

#### Placebo and falsification tests (from 9l)

| Test | Expected result | What it proves |
|---|---|---|
| Control occupation seniority shift | Null (no shift) | SWE shift is occupation-specific |
| Random time-split within Kaggle | Null | Our method doesn't find "change" in noise |
| Permuted period labels | Observed > 95th percentile of permuted | The difference exceeds random variation |
| SWE-adjacent (data scientist, PM) | Intermediate or null | Calibrates AI-exposure gradient |

#### Effect size reporting

Every test reports both statistical significance AND practical significance:

| Measure | When to use | "Small" | "Medium" | "Large" |
|---|---|---|---|---|
| Cramér's V | Chi-squared tests | 0.10 | 0.30 | 0.50 |
| Cohen's d | Continuous comparisons | 0.20 | 0.50 | 0.80 |
| Rank-biserial r | Mann-Whitney U | 0.10 | 0.30 | 0.50 |
| Log-odds ratio | Fightin' Words | 0.5 | 1.0 | 2.0 |
| Prevalence Δ (pp) | Skill prevalence shifts | 3pp | 5pp | 10pp |

A finding that is statistically significant (p < 0.05) but practically trivial (effect size below "small") is noted but not emphasized.

---

### Analysis outputs by RQ

| RQ | Primary table/figure | What it shows |
|---|---|---|
| **RQ1** | Table: Seniority distribution shift with chi-squared and Cramér's V | Junior share declined by X pp |
| **RQ1** | Figure: UMAP with 2024 seniority boundary, 2026 junior postings overlaid | Visual evidence of content convergence |
| **RQ1** | Table: Scope inflation metrics (5 measures, FDR-corrected) | Entry-level requirements inflated |
| **RQ2** | Figure: Skill migration heatmap (prevalence by seniority × period) | Which skills migrated |
| **RQ2** | Table: Top 15 migrated skills with Δ prevalence and FDR-adjusted p | Statistical proof of migration |
| **RQ2** | Figure: Skill co-occurrence network (2024 vs. 2026 junior) | Structural reorganization of junior skill requirements |
| **RQ3** | Figure: Revelio SWE openings with Bai-Perron breakpoints | External evidence of structural break |
| **RQ3** | Table: Permutation test results for cross-sectional magnitude | Our observed difference exceeds random variation |
| **RQ4** | Table: DiD regression coefficients for 4+ outcomes | SWE-specific effects isolated |
| **RQ4** | Figure: Embedding drift by occupation with bootstrap CIs | Visual evidence of SWE-specific change |
| **RQ6** | Table: Management vs. AI-orchestration keyword prevalence shift | Senior archetype changing |
| **RQ6** | Figure: Archetype Shift Index with bootstrap CI | Directional measure of transformation |
| **RQ7** | Table: Senior SWE archetype by computing era | Historical contextualization |
| **All** | Figure: Specification curve for each primary finding | Robustness across 324 specifications |
| **All** | Table: Placebo test results | Falsification evidence |

---


---

## Stage 12: Statistical verification

This stage systematically challenges every finding from Stage 11. The goal is to show that our results are not artifacts of sample composition, measurement error, random noise, or macro trends. A reviewer should be able to read this section and conclude: "They tried hard to break their own findings and couldn't."

Stage 12 runs AFTER the primary analysis (Stage 11) produces initial results. We design the verification battery now, but execute it only once we have findings to verify.

---

### 12a. Power analysis and sample size verification

**Why this comes first:** If we lack statistical power to detect a plausible effect, a null result is uninformative and a significant result may be inflated (winner's curse). We must confirm we have adequate power BEFORE interpreting results.

**Actual sample sizes (from data investigation):**

| Group | Kaggle (April 2024) | Scraped (March 2026) |
|---|---|---|
| SWE total | 3,466 | ~14,391 |
| SWE entry-level (native label) | ~385 | ~1,920 |
| SWE entry-level (after imputation, estimated) | ~600 | ~2,500 |
| SWE mid-senior | ~1,770 | ~7,300 |
| Control total | 10,766 | 17,246 |
| Nursing (largest control) | 6,082 | 5,360 |
| Civil eng | 157 | 547 |
| Mechanical eng | 265 | 1,388 |
| Accountant | 1,317 | 2,139 |

**Power calculations for each primary test:**

#### RQ1: Junior share shift (chi-squared)

```python
from statsmodels.stats.power import GofChisquarePower
power_analysis = GofChisquarePower()

# Minimum detectable effect: shift of 5 percentage points in junior share
# e.g., 12.3% → 7.3% or 12.3% → 17.3%
# n1 = 3,466 (Kaggle SWE), n2 = 14,391 (scraped SWE)
# Effect size w for chi-squared ≈ 0.05 for a 5pp shift in one cell of 3×2 table
mde = power_analysis.solve_power(effect_size=None, nobs=3466, alpha=0.05, power=0.80, n_bins=3)
print(f"Minimum detectable effect size (w): {mde:.4f}")
```

**Expected:** With n=3,466 in the smaller group, we have >80% power to detect a Cramér's V of ~0.05 (a 3-5pp shift in junior share). This is adequate for our hypothesized effect.

#### RQ1: Scope inflation (Mann-Whitney on description length)

```python
from statsmodels.stats.power import TTestIndPower
power = TTestIndPower()

# Kaggle entry-level SWE: ~600 (after imputation)
# Scraped entry-level SWE: ~2,500
# What's the minimum detectable Cohen's d?
mde = power.solve_power(effect_size=None, nobs1=600, ratio=2500/600,
                         alpha=0.05, power=0.80, alternative='two-sided')
print(f"Minimum detectable Cohen's d: {mde:.3f}")
```

**Expected:** With 600 vs. 2,500, we can detect Cohen's d ≈ 0.14 (a small effect). This is adequate — if scope inflation is real, the effect should be at least "small" by conventional standards.

#### RQ4: DiD interaction term

```python
# DiD requires adequate samples in all 4 cells:
# SWE × 2024: 3,466    SWE × 2026: 14,391
# Control × 2024: 10,766    Control × 2026: 17,246
# The binding constraint is the smallest cell: SWE × 2024 (3,466)
# With 3,466 in the smallest cell, detect interaction effect of ~0.05 SD
```

**Power table to report in methodology:**

| Test | Smaller group n | Minimum detectable effect | Adequate? |
|---|---|---|---|
| Junior share shift (chi-sq) | 3,466 | Cramér's V ≈ 0.05 (3-5pp shift) | Yes |
| Description length (M-W) | 600 entry-level | Cohen's d ≈ 0.14 | Yes |
| Skill prevalence per skill (z-test) | 600 entry-level | 4pp prevalence shift | Marginal — some rare skills may be underpowered |
| DiD interaction | 3,466 SWE × 2024 | β₃ ≈ 0.05 SD | Yes |
| Embedding drift (bootstrap) | 600 entry-level | cos distance ≈ 0.02 | Yes |
| Archetype Shift Index (bootstrap) | ~1,770 senior × 2024 | Ratio change ≈ 0.10 | Yes |

**Decision rule:** If a test is underpowered (power < 0.60 for the hypothesized effect), do NOT run it as a primary test. Report it as exploratory or aggregate to a coarser level (e.g., combine rare skills into skill clusters).

---

### 12b. Within-period placebo tests

**The core logic:** If our method detects a "change" between two halves of the same period (where no real change occurred), it's measuring noise, not signal. Every finding must pass this sanity check.

#### Placebo 1: Early vs. late Kaggle (within April 2024)

The Kaggle data spans ~4 weeks (April 5-20, 2024). Split into two halves:
- Early: before April 12 (30,101 rows, 743 SWE)
- Late: April 12+ (93,748 rows, 2,723 SWE)

Run the SAME tests from Stage 11 on this split:

| Test | Expected result | If NOT null |
|---|---|---|
| Junior share shift (chi-sq) | p > 0.05 (no shift) | Our method detects noise |
| Description length (M-W) | p > 0.05 | Scraping artifacts within Kaggle |
| Skill prevalence shifts (FDR) | 0 significant skills | Too many false positives |
| Content convergence (redefinition rate) | Rate ≈ baseline error rate | Classifier is unstable |

**The early/late split also controls for scraping order effects.** If the Kaggle scraper captured different types of jobs on different days (e.g., scraping tech companies first, then healthcare), the split would reveal this artifact.

**Sample size concern:** 743 SWE postings in the early split is small. If tests are underpowered at this sample size, compute the minimum detectable effect and report: "Our placebo test would have detected an effect of size X with 80% power. We observed null results, consistent with no within-period change, though small effects cannot be ruled out."

#### Placebo 2: Day-to-day variation in scraped data

We have 14 daily scrapes. Split into two arbitrary halves:
- Week 1: March 5-11 (7 days)
- Week 2: March 12-18 (7 days)

Run the same comparison between these two weeks. We expect NO significant differences — if we find them, it means daily variation is a concern and our results need date-level controls.

**Stronger version:** Run the full test battery on every possible 7-day vs. 7-day split (C(14,7) = 3,432 splits). Compute the distribution of test statistics under these permutations. Our cross-period test statistic should be far outside this null distribution.

```python
from itertools import combinations
import numpy as np

daily_indices = list(range(14))  # days 0-13
null_distribution = []
for combo in combinations(daily_indices, 7):
    week_a = postings_from_days(combo)
    week_b = postings_from_days(set(daily_indices) - set(combo))
    stat = compute_junior_share_diff(week_a, week_b)
    null_distribution.append(stat)

# Our cross-period statistic should exceed 99% of this null
cross_period_stat = compute_junior_share_diff(kaggle_swe, scraped_swe)
p_value = (np.abs(null_distribution) >= np.abs(cross_period_stat)).mean()
```

#### Placebo 3: Random subsample equivalence

Randomly split each period's SWE postings into two halves (A and B). Compare A₂₀₂₄ vs. B₂₀₂₄ and A₂₀₂₆ vs. B₂₀₂₆. Neither comparison should yield significant results. If they do, our sample has internal heterogeneity that inflates test statistics.

Run 100 random splits and report the fraction producing p < 0.05 for each test. Under the null, this fraction should be ~5%. If it's much higher, something is wrong (e.g., batch effects from daily scraping, company clustering, geographic concentration).

---

### 12c. Control group verification

**The logic:** Our primary findings claim SWE roles changed between 2024 and 2026. If control occupations (nursing, civil engineering, accounting) show the SAME changes, the finding is not SWE-specific — it's a platform change, a macro trend, or a scraping artifact.

#### Control group 1: Low-AI-exposure occupations (primary controls)

Run every Stage 11 test on control occupations:

| Test | SWE result | Control expected | If control matches SWE |
|---|---|---|---|
| Junior share shift | Significant decline | No change | Macro trend, not AI-driven |
| Scope inflation (desc. length, skill count) | Significant increase | No change | Scraping artifact |
| Content convergence | High redefinition rate | Low/baseline rate | Classifier drift |
| AI keyword emergence | Significant increase | No change (or much smaller) | Expected — this IS the differentiator |

**Per-occupation controls (run separately, then pool):**

| Occupation | Kaggle n | Scraped n | Adequate for DiD? |
|---|---|---|---|
| Nursing | 6,082 | 5,360 | Yes — large sample, strong control |
| Accountant | 1,317 | 2,139 | Yes |
| Electrical eng | 377 | 1,435 | Marginal |
| Financial analyst | 372 | 1,417 | Marginal |
| Mechanical eng | 265 | 1,388 | Marginal |
| Civil eng | 157 | 547 | Underpowered — report but caveat |
| Chemical eng | 27 | 34 | No — exclude from analysis |

**Pooled control:** Combine all control occupations (10,766 Kaggle + 17,246 scraped). This gives the best-powered DiD. Then run per-occupation to show results are consistent across controls.

**AI-exposure gradient test:** Order occupations by Felten et al. (2023) AI-exposure scores. If our effects scale with AI exposure (largest for SWE, intermediate for data scientists, small for accountants, null for nurses), that's strong evidence of an AI-driven mechanism.

```python
# Compute effect magnitude by occupation, plot against AI-exposure score
effects = {}
for occ in ['SWE', 'data_scientist', 'product_manager', 'accountant',
            'financial_analyst', 'mech_eng', 'nursing']:
    effects[occ] = compute_junior_share_shift(occ)

ai_exposure = get_felten_scores(effects.keys())
correlation = np.corrcoef(list(ai_exposure.values()),
                          list(effects.values()))[0,1]
# If correlation > 0.5 and significant, the effect scales with AI exposure
```

#### Control group 2: SWE-adjacent occupations (dose-response)

SWE-adjacent roles (data scientist, product manager, UX designer, QA engineer) are partially AI-exposed. If they show intermediate-sized effects (smaller than SWE, larger than controls), that's a dose-response pattern consistent with AI exposure being the mechanism.

| Occupation | AI exposure (Felten) | Expected effect |
|---|---|---|
| SWE | High | Large |
| Data scientist | High | Large (similar to SWE) |
| Product manager | Medium-high | Medium |
| UX designer | Medium | Small-medium |
| QA engineer | Medium | Medium |
| Accountant | Low | Small/null |
| Civil engineer | Low | Null |
| Nursing | Very low | Null |

**If the gradient is confirmed:** This is a powerful finding. It turns RQ4 from a binary question ("SWE-specific or not?") into a continuous relationship: the magnitude of restructuring scales with AI exposure.

#### Control group 3: Same-company cross-occupation

For companies appearing in both SWE and non-SWE postings within the same period, compare whether the company's SWE postings changed differently than its non-SWE postings. This holds employer-level factors constant (company culture, HR practices, scraping effects) and isolates the occupation-level effect.

```python
# Companies with both SWE and control postings in both periods
overlap_companies = (set(swe_2024['company']) & set(swe_2026['company'])
                    & set(ctrl_2024['company']) & set(ctrl_2026['company']))

for company in overlap_companies:
    swe_shift = junior_share(swe_2026[company]) - junior_share(swe_2024[company])
    ctrl_shift = junior_share(ctrl_2026[company]) - junior_share(ctrl_2024[company])
    within_company_did = swe_shift - ctrl_shift
```

This is the most demanding test: it asks whether the SAME EMPLOYER changed its SWE hiring differently than its non-SWE hiring.

---

### 12d. Alternative explanation tests

Each test here addresses a specific alternative explanation for our findings. If any alternative survives, we must either refute it or acknowledge it as a limitation.

#### Alternative 1: "It's just LinkedIn's algorithm/UI changing"

**Threat:** LinkedIn may have changed how it displays, ranks, or categorizes job postings between 2024 and 2026. A change in LinkedIn's seniority labeling algorithm would directly cause the seniority distribution shift we observe.

**Test:** Compare the distribution of LinkedIn's native `job_level` labels between periods (for LinkedIn-only data). If "not applicable" rate changed, or if LinkedIn relabeled categories, the shift is partly artificial.

Also check: did the ratio of seniority labels change for NON-SWE postings? If LinkedIn changed its labeling, it would affect all occupations equally. SWE-specific changes cannot be explained by platform changes.

```python
# Compare "not applicable" rate across periods and occupations
na_rate_swe_2024 = (swe_2024['seniority_native'] == 'not applicable').mean()
na_rate_swe_2026 = (swe_2026['seniority_native'] == 'not applicable').mean()
na_rate_ctrl_2024 = (ctrl_2024['seniority_native'] == 'not applicable').mean()
na_rate_ctrl_2026 = (ctrl_2026['seniority_native'] == 'not applicable').mean()
# If SWE NA rate changed but control didn't → NOT a platform change
```

#### Alternative 2: "It's a composition effect (different companies)"

**Threat:** The 2026 sample might capture a different mix of companies (more FAANG, fewer startups, or vice versa) with inherently different seniority distributions.

**Test:** The Oaxaca-Blinder decomposition from 9g. Additionally:

```python
# Within overlapping companies: does the shift hold?
shared_companies = set(swe_2024['company_normalized']) & set(swe_2026['company_normalized'])
within_shift = (
    swe_2026[swe_2026['company_normalized'].isin(shared_companies)]['seniority']
    .value_counts(normalize=True) -
    swe_2024[swe_2024['company_normalized'].isin(shared_companies)]['seniority']
    .value_counts(normalize=True)
)
# If the shift holds within shared companies, it's not composition
```

**Report:** "X% of the observed shift is explained by composition changes; Y% remains after controlling for company overlap."

#### Alternative 3: "It's the scraping methodology"

**Threat:** Our scraper captures different postings than the Kaggle scraper. Query design, pagination depth, geographic scope, and anti-bot handling all differ. The observed differences may reflect what each scraper captures, not what the labor market looks like.

**Tests:**
1. **Indeed-only vs. LinkedIn-only:** Run the analysis separately on each platform within the scraped data. If both platforms show the same SWE shift, it's unlikely to be a single-platform scraping artifact.
2. **Geographic subset test:** Restrict both datasets to the same top-5 states (CA, TX, WA, VA, NY — present in both). If the shift holds in matched geography, geographic coverage differences aren't driving it.
3. **Company-matched test:** Use only companies appearing in both datasets. This is the strongest control for scraping methodology — the same companies, captured by different scrapers.

#### Alternative 4: "It's seasonal (April vs. March)"

**Threat:** April and March have different hiring patterns. Spring budget cycles, new-year planning, etc.

**Test:** Pull JOLTS monthly data for the information sector. Compare March vs. April historically (2019-2024). If the March-April difference in job openings is typically <3%, seasonality cannot explain a 5+pp shift in junior share.

```python
jolts_info = jolts[jolts['series'] == 'Information']
jolts_info['month'] = jolts_info['date'].dt.month
march = jolts_info[jolts_info['month'] == 3]['value']
april = jolts_info[jolts_info['month'] == 4]['value']
seasonal_diff = ((april.values - march.values) / march.values).mean()
print(f"Average March→April seasonal change: {seasonal_diff:.1%}")
```

#### Alternative 5: "Ghost jobs are driving the results"

**Threat:** If ghost job prevalence increased between 2024 and 2026 (CRS estimates 12.5% in 2023 → ~20% in 2025), and ghost jobs disproportionately inflate entry-level requirements, our "scope inflation" finding could partly reflect ghost job growth rather than genuine restructuring.

**Test:** Run the full analysis excluding postings flagged as high ghost-job risk (from Stage 8a). If findings hold after exclusion, ghost jobs aren't driving them. Report results both with and without ghost-flagged postings.

---

### 12e. Sensitivity analyses

Each sensitivity analysis asks: "If we made a different reasonable methodological choice, would we reach the same conclusion?"

#### Seniority classifier sensitivity

Run all primary tests under 3 classification schemes:
1. Our rule-based imputer (default)
2. LinkedIn native labels where available, imputed only where missing
3. SetFit description-based classifier (from Stage 5a)

Report a table: do all 3 classifiers produce the same directional finding? If yes, the finding is robust to classifier choice. If not, identify which postings are classified differently and investigate.

#### SWE definition sensitivity

Run under 3 SWE definitions:
1. Narrow: core SWE titles only (excluding data engineer, ML engineer)
2. Standard: our canonical SWE_PATTERN (default)
3. Broad: include data scientist, QA engineer, product engineer

If the finding holds only under the broad definition, it's driven by non-core SWE roles and needs nuancing.

#### Deduplication sensitivity

Run under 3 dedup levels:
1. No near-dedup (keep all postings with unique URLs)
2. Standard near-dedup (cosine ≥ 0.70)
3. Aggressive dedup (cosine ≥ 0.50, collapses multi-location)

#### Platform sensitivity

Run on:
1. LinkedIn only (both periods on the same platform)
2. Full sample (LinkedIn + Indeed for 2026)

If findings differ between LinkedIn-only and full-sample, the Indeed inclusion is driving partial results. LinkedIn-only is the more conservative and defensible comparison.

#### Company concentration sensitivity

Run with:
1. Full sample
2. Capped at 10 postings per company
3. Excluding top-5 companies from each dataset
4. Excluding DataAnnotation specifically (crowdwork platform in Kaggle)

#### Geographic sensitivity

Run on:
1. Full sample
2. Top-5 metros only (matched across datasets)
3. Excluding remote postings

#### Produce a consolidated sensitivity table

| Finding | Default | Alt seniority | Alt SWE def | LinkedIn-only | Cap 10/co | Top-5 metros | Verdict |
|---|---|---|---|---|---|---|---|
| Junior share ↓ | p=X, Δ=Y | p=X, Δ=Y | ... | ... | ... | ... | Robust/Suggestive/Fragile |
| Scope inflation | ... | ... | ... | ... | ... | ... | ... |
| Skill migration (top skill) | ... | ... | ... | ... | ... | ... | ... |
| DiD β₃ | ... | ... | ... | ... | ... | ... | ... |
| Archetype shift | ... | ... | ... | ... | ... | ... | ... |

**Verdict rules:**
- **Robust:** Significant (p < 0.05) and same direction in ≥ 5 of 6 sensitivity variants
- **Suggestive:** Significant and same direction in 3-4 of 6 variants
- **Fragile:** Significant in < 3 variants → not reported as a finding

---

### 12f. Effect size calibration

**Why this is separate from significance testing:** A statistically significant result with a tiny effect size is practically meaningless. Conversely, a large effect size with p = 0.06 is still informative. We report both, but calibrate what effect sizes mean in context.

#### External benchmarks for calibration

| Metric | Our observed Δ | Context benchmark | Interpretation |
|---|---|---|---|
| Junior share change | X pp | Hershbein & Kahn (2018): junior share dropped ~5pp during Great Recession | If our Δ is similar magnitude, it's comparable to a recession-level shock |
| Description length change | X words | Deming & Kahn (2018): skill requirements increased ~20% 2007-2017 | This gives a decade-scale baseline for scope inflation |
| AI keyword emergence | X pp | Acemoglu et al. (2022): AI-exposed occupations saw 15-20% task reallocation | Calibrates what "meaningful" AI adoption looks like |
| Embedding drift (cosine) | X | Need to establish: what's the typical within-period drift? | If cross-period drift >> within-period drift, the change is real |

#### Practical significance thresholds

For this study, we define practically significant as:
- **Junior share change:** ≥ 3 percentage points (roughly the MDE given our power)
- **Skill prevalence change:** ≥ 5 percentage points for any individual skill
- **Description length change:** ≥ 15% increase in requirements section
- **Redefinition rate:** ≥ 10 percentage points above the baseline misclassification rate
- **DiD interaction:** ≥ 0.10 SD (small-to-medium effect)
- **Archetype Shift Index change:** ≥ 20% relative increase

These thresholds are set before looking at results (pre-registered). Findings that cross both statistical and practical significance thresholds are "strong evidence." Findings that are statistically significant but below practical thresholds are "detectable but modest."

---

### 12g. Reproducibility protocol

**Why this matters:** Computational social science papers are notoriously difficult to replicate. Pin every random seed, model version, and data transform.

1. **Random seeds:** Set `np.random.seed(42)` and `torch.manual_seed(42)` at the start of every analysis script. Report the seed.
2. **Model versioning:** Pin all HuggingFace model versions by commit hash (not just model name). E.g., `TechWolf/JobBERT-v2` at commit `abc123`.
3. **Data versioning:** Hash the unified.parquet file (SHA-256) and report in methodology. Any re-run must produce the same hash after preprocessing.
4. **Pipeline code:** All preprocessing and analysis code in version-controlled scripts. No manual steps between preprocessing and results.
5. **Intermediate outputs:** Cache and version all embeddings, topic models, and classifier predictions. A reviewer should be able to start from cached embeddings and reproduce all results without re-running the embedding step.

---

### Verification outputs

| Output | Type | Paper section |
|---|---|---|
| Power analysis table | Table | §3.5 Statistical power |
| Within-period placebo results (3 tests) | Table | §4.4 Placebo tests |
| Control group results (per occupation + pooled) | Table + figure | §4.3 Control analysis |
| AI-exposure gradient plot | Figure | §4.3 (if gradient confirmed) |
| Alternative explanation tests (5 tests) | Table | §4.4 or §5 Discussion |
| Sensitivity analysis matrix | Table (consolidated) | §4.5 Sensitivity |
| Specification curve plots (per finding) | Figures | §4.5 or Appendix |
| Effect size calibration table | Table | §4.2 alongside main results |
| Reproducibility metadata | Appendix | Appendix C |

---


---

## Implementation order

### Phase 5: Formal Analysis (Stage 11)

```
33. RQ1 seniority shift + content convergence + scope inflation (11a)
34. RQ2 skill prevalence shifts + Fightin' Words + co-occurrence networks (11b)
35. RQ3 external breakpoint detection + permutation magnitude test (11c)
36. RQ4 DiD + synthetic control + embedding drift comparison (11d)
37. RQ6 management vs AI-orchestration keyword shift + BERTopic (11e)
38. RQ5/RQ7 qualitative synthesis + historical comparison (11f)
39. Specification curves + multiple testing corrections + bootstrap CIs (11g)
```

### Phase 6: Statistical Verification (after Stage 11 produces results)

```
40. Power analysis for every primary test (12a)
41. Within-period placebos: early/late Kaggle, week 1/2 scraped, random splits (12b)
42. Control group verification: per-occupation + pooled + AI-exposure gradient (12c)
43. Alternative explanation tests: platform, composition, scraping, seasonality, ghost (12d)
44. Sensitivity matrix: seniority × SWE def × dedup × platform × company × geography (12e)
45. Effect size calibration against external benchmarks (12f)
46. Reproducibility protocol: seeds, model versions, data hashes (12g)
```
