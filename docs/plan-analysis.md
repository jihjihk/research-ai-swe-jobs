# Analysis Plan

Date: 2026-03-20
Status: Draft — ready for review before implementation

This document covers Stage 15 (formal analysis for RQ1-RQ4) and Stage 16 (statistical verification and robustness). It runs after preprocessing (Stages 1-12 in `plan-preprocessing.md`) and exploration/validation (Stages 13-14 in `plan-exploration.md`) are complete.

**Research questions:** RQ1 (employer-side restructuring), RQ2 (task and requirement migration), RQ3 (employer-requirement / worker-usage divergence), RQ4 (mechanisms). See `docs/1-research-design.md` for full specifications.

**Design constraint:** We have two cross-sections (April 2024, March 2026), not a panel. All analyses are framed as "consistent with" structural change, not as causal proof.

---

## Data sources

| Source | Rows | Date | SWE postings | Entry-level labels | Platform |
|---|---|---|---|---|---|
| Kaggle arshkon | 124K | April 2024 | ~3,466 | ~385 native entry-level SWE | LinkedIn |
| Kaggle asaniczka | 1.35M | January 2024 | ~18,169 US SWE | NO entry-level labels (only "Mid senior" and "Associate") | LinkedIn |
| Scraped | ~3,680 SWE/day + ~30,888 non-SWE/day | March 2026+ | ~14,391 unique | Via classifier | LinkedIn (primary), Indeed (sensitivity) |

**Primary platform:** LinkedIn only. Indeed is used for sensitivity analyses.

**Major constraint:** The asaniczka dataset lacks entry-level seniority labels. Entry-level analysis relies primarily on the arshkon dataset (~385 native entry-level SWE) supplemented by seniority imputation.

**Dataset outputs:**
- `unified.parquet`: canonical postings corpus
- `unified_observations.parquet`: daily observation panel
- Measurement appendix documenting dedupe, cleaning, and construct definitions

---

## Empirical strategy

The analysis follows the empirical strategy defined in `1-research-design.md`:

1. **Descriptive restructuring first** — backbone of the paper
2. **Paired historical comparison** — 2024 Kaggle vs 2026 scraped
3. **Break analysis as supportive evidence** — not the backbone
4. **Comparative benchmarking** — compare findings to Brynjolfsson, Acemoglu, Hampole; do not replicate
5. **Sensitivity analyses** — specification curves across data and method variants

---

## Key constructs

### Junior scope inflation

Entry-level or junior-labeled postings increasingly asking for requirements that historically clustered in mid-level or senior roles.

Candidate indicators:

- higher required years of experience within junior-tagged postings
- more system-design, ownership, and architecture language in junior postings
- higher junior-to-senior embedding similarity over time

### Senior archetype shift

Senior postings moving from people-management and team-development language toward review, architecture, AI-enabled leverage, and orchestration language.

Candidate indicators:

- decline in management keywords
- rise in orchestration / review / agent / evaluation language
- shifts in skill and task bundles within senior postings

### Posting-usage divergence

The gap between employer-side AI requirements and observed AI usage in comparable occupation groups.

Framed as an **employer-requirement / worker-usage divergence index**, not a direct treatment effect.

External benchmarks: Anthropic occupation-level AI usage data, Stack Overflow Developer Survey.

### Ghost requirements

Requirements listed in postings that hiring-side actors describe as aspirational, template-driven, defensive, or not meaningfully screened in practice.

Validated qualitatively through interviews (RQ4), not inferred from text alone.

---

## Stage 15: Formal analysis

This is the core analysis that produces the paper's findings. Each RQ gets a primary test (the simplest credible method that answers the question), stronger tests (ML/NLP-powered, higher power or richer signal), and robustness checks. Every test is paired with the specific alternative finding that would falsify the hypothesis.

---

### 15a. RQ1: Employer-side restructuring

**Question:** How did employer-side SWE requirements restructure across seniority levels from 2023 to 2026?

**What we need to show:** Changes in (a) junior posting share and volume, (b) junior scope inflation, (c) senior role redefinition, and (d) source-specific and metro-specific heterogeneity.

**Null hypothesis:** The seniority distribution, content profile of junior postings, and archetype composition of senior postings are statistically indistinguishable across the two periods, after controlling for composition.

#### Test 1: Seniority distribution shift (primary)

Compare the proportion of entry-level / associate / mid-senior SWE postings between April 2024 and March 2026.

```
H0: P(junior | SWE, 2024) = P(junior | SWE, 2026)
H1: P(junior | SWE, 2024) != P(junior | SWE, 2026)
```

**Method:** Chi-squared test of homogeneity on the 3x2 contingency table (3 seniority levels x 2 periods). Report Cramer's V for effect size.

```python
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df['period'], df['seniority_3level'])
chi2, p, dof, expected = chi2_contingency(contingency)
cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
```

**Sufficient evidence:** p < 0.05 with Cramer's V > 0.05 (small-but-meaningful effect). Report the actual shift in junior share (percentage points) with bootstrap 95% CI.

**What would falsify:** If the junior share is unchanged or increased, the "narrowing" narrative is wrong. If the distribution shifted but only because a different mix of companies was captured, it is a composition artifact (checked in sensitivity analyses).

#### Test 2: Content convergence — the redefinition test (stronger)

Separates title inflation (cosmetic relabeling) from genuine content change. A 2026 "Junior SWE" posting that reads like a 2024 "Mid-Senior SWE" posting is a redefined role, not a disappeared one.

**Method:** Train a logistic regression seniority classifier on 2024 JobBERT-v2 embeddings. Apply it to 2026 postings. Measure the "redefinition rate": the fraction of 2026 entry-level postings that the 2024-trained model predicts as mid-senior.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(embeddings_2024, seniority_2024)

junior_2026 = df_2026[df_2026['seniority'] == 'entry level']
preds = clf.predict(embeddings_junior_2026)
redefinition_rate = (preds != 'junior').mean()
```

**Sufficient evidence:** If the redefinition rate for 2026 junior postings is significantly higher than the misclassification rate on 2024 junior postings (baseline error rate). Compare with a permutation test: shuffle the period labels 1000 times and recompute. The observed redefinition rate should exceed 95% of permuted rates.

**Alternative method (model-free):** Compute cosine similarity between each 2026 junior posting embedding and the 2024 junior centroid vs. the 2024 senior centroid. If more 2026 junior postings are closer to the senior centroid than to the junior centroid (compared to 2024 junior postings), that is convergence. Test with a two-sample t-test on the similarity ratio.

#### Test 3: Scope inflation metrics

Run on junior postings only, comparing 2024 vs. 2026:

| Metric | Test | Interpretation |
|---|---|---|
| Description length (requirements section only) | Mann-Whitney U | Longer requirements = more demanded of juniors |
| Distinct skill count per posting | Mann-Whitney U | More skills = broader scope |
| Years of experience required (median) | Mann-Whitney U | Higher YoE = inflated requirements |
| Skill Breadth Index (ESCO-mapped distinct skills) | Mann-Whitney U | Taxonomy-grounded scope measure |
| Senior keyword infiltration rate | Proportion test (z-test) | "system design", "architecture" appearing in junior postings |

Use Benjamini-Hochberg FDR correction across the battery. Report effect sizes (Cohen's d or rank-biserial correlation) alongside p-values.

#### Test 4: Senior archetype shift — keyword prevalence (primary)

Define two keyword categories grounded in the research design:

**Management category:** mentorship, coaching, hiring, team lead, team leadership, performance review, people management, direct reports, staff development, career development, 1:1, one-on-one

**AI-orchestration category:** AI agent, LLM, large language model, prompt engineering, RAG, retrieval augmented, model evaluation, AI integration, copilot, AI orchestration, AI-assisted, agent framework, vector database, agentic

For each category, compute prevalence (fraction of senior SWE postings mentioning at least one keyword) in each period. Test with two-proportion z-test.

**Archetype Shift Index:** Compute the ratio (AI-orchestration prevalence) / (management prevalence) for each period. This single number captures the directional shift. Test whether it increased with a bootstrap CI.

#### Test 5: Senior archetype shift — Fightin' Words

Run Fightin' Words comparing senior 2024 vs. senior 2026 descriptions. The top terms distinguish the two periods without requiring predefined keyword lists. This checks whether our keyword categories are well-constructed: if our chosen keywords rank highly in the Fightin' Words output, the categories are valid. If other terms rank higher, we may be missing important dimensions.

#### Test 6: Senior archetype shift — BERTopic

Run BERTopic separately on senior SWE postings (both periods combined). Examine whether:

- management-themed topics are more prevalent in 2024 than 2026
- AI/technical-themed topics are more prevalent in 2026 than 2024
- new topics exist in 2026 that do not appear in 2024 (emerging senior archetype)

Use `topics_over_time()` with 2 bins (2024, 2026) to quantify topic prevalence shift. This is exploratory/robustness, not headline evidence.

#### Test 7: Controlled regressions

**Junior scope inflation (junior postings only):**

```
SkillBreadth_i = b0 + b1(Period2026) + b2(CompanySize) + b3(Industry)
                + b4(Metro) + b5(IsRemote) + e_i
```

b1 is the period effect after controlling for composition. HC3 robust standard errors. Cluster by company if enough firms appear in both periods.

**Senior archetype shift (senior postings only):**

```
MgmtKeywordCount_i = b0 + b1(Post2026) + b2(CompanySize) + b3(Industry) + e
AIKeywordCount_i   = b0 + b1(Post2026) + b2(CompanySize) + b3(Industry) + e
```

b1 should be negative for management and positive for AI-orchestration. HC3 standard errors.

#### Test 8: Source-specific and metro-specific heterogeneity

Run all primary tests separately by:

- data source (Kaggle arshkon vs. scraped LinkedIn)
- metro area (top 5 metros individually)
- company size tier (small / medium / large)

Report heterogeneity as a table. If results are directionally consistent across subgroups, the finding is not driven by a single source or metro. If results diverge, report the divergence and investigate.

**Sufficient evidence for RQ1 overall:** At least 3 of the test families (seniority shift, content convergence, scope inflation, senior archetype shift) pointing in the same direction. Heterogeneity analysis showing directional consistency across sources and metros. A single test alone is not conclusive given our data limitations.

---

### 15b. RQ2: Task and requirement migration

**Question:** Which requirements moved downward into junior postings, and which senior-role responsibilities shifted from management toward AI-enabled orchestration?

**What we need to show:** Specific skills and requirements that were predominantly senior-associated in 2024 appear at significantly higher rates in junior postings by 2026.

**Null hypothesis:** The skill profile of junior postings is unchanged between periods. No individual skill shows a statistically significant increase in junior prevalence.

**Primary focus areas:** system design, CI/CD and deployment ownership, cross-functional coordination, end-to-end ownership, AI-tool proficiency, mentorship / hiring / team-lead language.

#### Test 1: Skill prevalence shift (primary)

For each ESCO-mapped skill, compute its prevalence in junior postings in each period. Test for significant changes.

```
H0: P(skill_k | junior, 2024) = P(skill_k | junior, 2026)   for each skill k
H1: P(skill_k | junior, 2024) != P(skill_k | junior, 2026)   for at least some k
```

**Method:** Two-proportion z-test for each skill. Apply Benjamini-Hochberg FDR at q = 0.05. Report adjusted p-values and the absolute prevalence change (pp).

**Visualization:** Skill migration heatmap. Rows = skills (sorted by change magnitude), columns = (junior 2024, senior 2024, junior 2026, senior 2026). Color = prevalence. The visual pattern of skills "sliding" from senior-only to junior+senior is the core RQ2 figure (Figure 5 in paper outputs).

#### Test 2: Fightin' Words for junior vocabulary shift

Run Fightin' Words (log-odds-ratio with Dirichlet prior) on:

- Junior 2024 vs. Junior 2026 (what changed in junior postings?)
- Junior 2026 vs. Senior 2024 (do 2026 juniors sound like 2024 seniors?)

Words with high positive log-odds in the second comparison AND high positive log-odds in the first comparison are "migrated" terms: newly associated with junior roles and resembling what senior roles used to require.

This is a high-value early analysis tool per `6-methods-learning.md`. Run it early and use the results to refine dictionaries and annotation rules.

#### Test 3: Skill co-occurrence network shift

Build a skill co-occurrence graph for junior postings in each period. Nodes = ESCO skills, edges = co-occurrence within the same posting, edge weights = PMI (pointwise mutual information).

Measure:

- new edges in 2026 that did not exist in 2024 (new skill combinations emerging in junior roles)
- skills that gained centrality (degree, betweenness) — hub requirements
- community structure changes — did junior skill clusters reorganize?

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

### 15c. RQ3: Employer-requirement / worker-usage divergence

**Question:** Do employer-side AI requirements outpace observed workplace AI usage, consistent with anticipatory restructuring?

This is a comparison between two different objects:

- employer-side posting requirements (from our corpus)
- worker-side observed AI usage or coverage (from external benchmarks)

The contribution is the divergence itself, not a claim that the two measures are directly interchangeable.

#### External benchmarks

| Source | What it measures | Level of analysis |
|---|---|---|
| Anthropic occupation-level AI usage data | Observed AI capability coverage by occupation | Occupation-level |
| Stack Overflow Developer Survey | Self-reported AI tool usage among developers | Individual developer-level |
| Brynjolfsson et al. AI adoption estimates | Firm-level AI adoption rates | Firm-level |
| Acemoglu et al. task exposure measures | AI exposure by occupation and task | Task-level |

#### Test 1: Divergence index construction

Compute an **employer-requirement / worker-usage divergence index**:

1. From our corpus: measure the fraction of SWE postings that mention AI tools, AI-assisted workflows, or AI proficiency requirements, by period.
2. From external benchmarks: extract the corresponding observed AI usage rate for SWE-equivalent occupations.
3. Compute the gap: (employer-side AI requirement rate) - (worker-side AI usage rate).
4. Track the gap over time (2024 vs. 2026) using our two cross-sections and the benchmark time series.

**Interpretation:** If the gap is widening (employer requirements growing faster than observed usage), this is consistent with anticipatory restructuring. If the gap is stable or narrowing, employers are tracking actual usage, not running ahead of it.

#### Test 2: Within-posting AI requirement analysis

Classify AI mentions in postings into categories:

- **Tool proficiency:** mentions of specific AI tools (GitHub Copilot, ChatGPT, Claude, etc.)
- **Workflow integration:** AI-assisted development, AI-powered testing, AI code review
- **Strategic/architectural:** AI system design, LLM integration, agent orchestration
- **Generic/aspirational:** "experience with AI", "AI enthusiasm", no specific tool or workflow

The ratio of generic/aspirational to specific/operational AI mentions is itself a signal: high generic-to-specific ratios suggest anticipatory posting rather than operational requirements.

#### Test 3: Cross-occupation AI requirement comparison

Compare AI requirement rates in SWE postings vs. control occupations. If SWE postings mention AI requirements at rates far exceeding what external benchmarks suggest for actual SWE AI usage, while control occupations do not show the same divergence, the pattern is SWE-specific.

#### Test 4: Temporal comparison with benchmark surveys

Where benchmark data has its own time dimension (e.g., Stack Overflow surveys across years), plot the employer-side requirement trajectory alongside the worker-side usage trajectory. The divergence plot (Figure 6 in paper outputs) is the core visual for RQ3.

**Sufficient evidence for RQ3:** A measurable and growing gap between employer-side AI requirements and external benchmark AI usage rates, with the gap larger for SWE than for control occupations.

---

### 15d. RQ4: Mechanisms (qualitative)

**Question:** How do senior engineers, junior engineers, and hiring-side actors explain the restructuring of SWE postings?

This is NOT a statistical test. It is an interview-based qualitative analysis using reflexive thematic analysis. Protocol details are in `docs/2-interview-design-mechanisms.md`.

#### Analytic method

Reflexive thematic analysis (Braun and Clarke 2006, 2019) with hybrid deductive/inductive coding.

**Deductive code families** (derived from constructs):

- `junior_scope_inflation` — do interviewees report that junior expectations have risen?
- `senior_archetype_shift` — do interviewees report senior work shifting from mentoring to orchestration?
- `ghost_requirement` — do interviewees describe requirements as aspirational or not screened?
- `jd_authorship` — who wrote or changed the job description, and why?
- `screened_vs_unscreened_requirement` — which listed requirements are actually evaluated?
- `actual_ai_workflow_change` — do interviewees describe real changes to daily work from AI?
- `anticipatory_restructuring` — are employers reacting to current AI or to expectations about future AI?

Inductive codes emerge from the data during analysis. Analytic memos after each interview. Cross-cohort comparison across the three cohorts (senior SWEs, junior SWEs, hiring-side actors).

#### What the interviews adjudicate

The interviews test whether observed posting changes reflect:

1. **Real workflow change** — AI tools genuinely changed what junior and senior engineers do
2. **HR template inflation** — job descriptions grew because HR departments added requirements without manager input
3. **Hiring-market overscreening** — a looser labor market let employers ask for more
4. **Anticipatory restructuring** — employers are positioning for where AI is going, not where it is today

These four mechanisms are not mutually exclusive. The qualitative contribution is determining their relative weight and how they interact.

#### Integration with quantitative findings

- Present quantitative findings (from RQ1-RQ3) to interviewees as elicitation material
- Use interview responses to validate or challenge the constructs measured quantitatively
- Ghost requirements can only be validated qualitatively — text analysis alone cannot determine whether a requirement is screened
- The hiring-side cohort is essential for adjudicating between mechanisms

#### LLM role in qualitative analysis

Per `6-methods-learning.md`: LLMs are annotation assistants, not qualitative researchers. Use LLMs only for:

- first-pass deductive coding after human codebook is defined
- proposing candidate codes for human review
- summarizing clusters of near-duplicate excerpts

Do not use LLMs for:

- generating final themes
- deciding whether a requirement is genuinely "ghost"
- replacing close reading on edge cases

**Output:** Interview sample and mechanism summary table (Table 6 in paper outputs).

---

### 15e. Supporting analyses

These are not primary RQ tests but provide supporting evidence and context.

#### Break analysis as supportive evidence

Use endogenous break detection and event-study style plots. Frame carefully: the design does not assume one universal "post-agent" date.

Candidate release windows to annotate:

- `2024-05-13` GPT-4o
- `2024-06-20` Claude 3.5 Sonnet
- `2025-05-22` Claude 4

**Method 1: External time series (Revelio + JOLTS)**

Run Bai-Perron endogenous breakpoint detection on the Revelio SOC-15 monthly job openings series.

```python
import ruptures as rpt

signal = revelio_soc15_openings.values
algo = rpt.Pelt(model="rbf", min_size=3).fit(signal)
breakpoints = algo.predict(pen=10)
```

Workflow:

1. UDmax test: any breaks at all in the Revelio SWE openings series?
2. If yes, estimate break date(s) with confidence intervals
3. Run the same on control occupation series (nursing, civil engineering). If they show the same break, it is macro, not SWE-specific.

**Method 2: Magnitude-of-change test**

Permutation test. Pool all SWE postings from both periods. Randomly assign to "2024" and "2026" groups (maintaining original group sizes). Recompute the seniority shift metric 10,000 times. The observed shift should exceed 95% of permuted shifts.

**Method 3: Multivariate change detection (BOCPD on external data)**

Run Bayesian Online Change Point Detection simultaneously on multiple Revelio/JOLTS series (SWE openings + hiring rate + salary trend + average skill count). Multivariate detection has higher power because it pools coordinated signals.

**Output:** Annotated break-analysis plot with candidate release windows (Figure 8 in paper outputs).

#### SWE-specific difference-in-differences (sensitivity analysis)

This is the old RQ4 from the prior plan, now demoted to a sensitivity analysis. It tests whether the patterns are SWE-specific or part of a broader trend.

```
Y_i = a + b1(SWE) + b2(Post2026) + b3(SWE x Post2026) + gX + e
```

b3 is the SWE-specific change beyond what control occupations experienced.

Run on multiple outcomes: junior share, skill breadth index, description length, AI-keyword prevalence. Use FDR correction across outcomes.

**Control occupations:** Use Felten et al. (2023) AI exposure scores. Select bottom-quartile occupations: civil engineering (SOC 17-2051), mechanical engineering (17-2141), registered nursing (29-1141), accounting (13-2011).

For the default production dataset, controls do not receive default LLM extraction or default control-wide LLM seniority classification. Cross-occupation analyses that need LLM-cleaned control text should use a separate control-extraction sensitivity run; control seniority remains a separate sensitivity question rather than part of the default Stage 9 classification path.

**Critical assumption:** Parallel pre-trends. With only 2 time points, we cannot directly test this. Mitigation: use external data (Revelio) to show SWE and controls were on parallel trajectories before 2024; report sensitivity of b3 to different control sets.

**AI-exposure gradient test:** Order occupations by AI-exposure scores. If effects scale with AI exposure (largest for SWE, intermediate for data scientists, small for accountants, null for nurses), that is strong evidence of an AI-driven mechanism.

#### Cross-occupation embedding drift

Measure whether SWE postings drifted more than control postings between periods.

```python
for occupation in ['SWE', 'civil_eng', 'nursing', 'mech_eng']:
    centroid_2024 = embeddings[occ == occupation & period == 2024].mean(axis=0)
    centroid_2026 = embeddings[occ == occupation & period == 2026].mean(axis=0)
    drift = 1 - cosine_similarity([centroid_2024], [centroid_2026])[0][0]
```

If SWE drift >> control drift, the change is SWE-specific. Test with bootstrap CIs on the drift difference.

---

### 15f. Robustness and multiple testing protocol

This section applies to ALL RQs. Without it, a reviewer can dismiss any individual finding as fragile or cherry-picked.

#### Specification curve analysis

Run every primary test under all defensible specifications. The specification space:

| Dimension | Variants | Count |
|---|---|---|
| SWE definition | Narrow, standard, broad | 3 |
| Seniority classifier | LLM-augmented, rule-based only, native labels only | 3 |
| Dedup threshold | Strict (exact), standard (0.70), loose (0.50) | 3 |
| Sample scope | Full, LinkedIn-only, excl. aggregators, top-10 metros | 4 |
| Company capping | None, cap at 10 per company, cap at median | 3 |

Total: 3 x 3 x 3 x 4 x 3 = 324 specifications per test.

**Rule:** A finding is "robust" if it is significant (p < 0.05) in > 80% of specifications. If 50-80%, it is "suggestive." If < 50%, it is "fragile" and not reported as a finding.

#### Multiple testing corrections

| Scope | Method | Rationale |
|---|---|---|
| Within a single RQ (e.g., testing 15 skills in RQ2) | Benjamini-Hochberg FDR at q = 0.05 | Controls false discovery rate |
| Across RQs (e.g., RQ1 + RQ2 + RQ3 primary tests) | Holm-Bonferroni | More conservative for headline findings |
| Specification curve | Report the fraction significant, not individual p-values | The curve itself accounts for multiplicity |

#### Bootstrap confidence intervals

For every key estimate (junior share change, skill prevalence change, Archetype Shift Index, DiD coefficient, divergence index), report bootstrap 95% CIs alongside the point estimate. Use 2000 bootstrap resamples with BCa (bias-corrected and accelerated) intervals.

```python
from scipy.stats import bootstrap
result = bootstrap((data,), np.mean, n_resamples=2000,
                   confidence_level=0.95, method='BCa')
```

#### Placebo and falsification tests

| Test | Expected result | What it proves |
|---|---|---|
| Control occupation seniority shift | Null (no shift) | SWE shift is occupation-specific |
| Random time-split within Kaggle | Null | Our method does not find "change" in noise |
| Permuted period labels | Observed > 95th percentile of permuted | The difference exceeds random variation |
| SWE-adjacent (data scientist, PM) | Intermediate or null | Calibrates AI-exposure gradient |

#### Effect size reporting

Every test reports both statistical significance AND practical significance:

| Measure | When to use | "Small" | "Medium" | "Large" |
|---|---|---|---|---|
| Cramer's V | Chi-squared tests | 0.10 | 0.30 | 0.50 |
| Cohen's d | Continuous comparisons | 0.20 | 0.50 | 0.80 |
| Rank-biserial r | Mann-Whitney U | 0.10 | 0.30 | 0.50 |
| Log-odds ratio | Fightin' Words | 0.5 | 1.0 | 2.0 |
| Prevalence change (pp) | Skill prevalence shifts | 3pp | 5pp | 10pp |

A finding that is statistically significant (p < 0.05) but practically trivial (effect size below "small") is noted but not emphasized.

---

### Analysis outputs by RQ

| RQ | Primary table/figure | What it shows |
|---|---|---|
| **RQ1** | Figure 1: Junior posting share and volume over time | Junior share and volume trends |
| **RQ1** | Figure 2: Junior scope-inflation index over time | Scope inflation trajectory |
| **RQ1** | Figure 3: Senior archetype shift index over time | Archetype shift trajectory |
| **RQ1** | Figure 4: Junior-senior embedding similarity over time | Content convergence |
| **RQ1** | Table 1: Summary statistics by source, period, and seniority | Descriptive foundation |
| **RQ1** | Table 3: Regression estimates for junior scope inflation | Controlled scope inflation |
| **RQ1** | Table 4: Regression estimates for senior archetype shift | Controlled archetype shift |
| **RQ1** | Table: Scope inflation metrics (5 measures, FDR-corrected) | Entry-level requirements inflated |
| **RQ1** | Table: Seniority distribution shift with chi-squared and Cramer's V | Junior share change |
| **RQ2** | Figure 5: Requirement migration heatmap by seniority and period | Which requirements migrated |
| **RQ2** | Table: Top 15 migrated skills with prevalence change and FDR-adjusted p | Statistical evidence of migration |
| **RQ2** | Figure: Skill co-occurrence network (2024 vs. 2026 junior) | Structural reorganization of junior skill requirements |
| **RQ3** | Figure 6: Employer-requirement / worker-usage divergence plot | Divergence trajectory |
| **RQ3** | Table: AI requirement classification (tool/workflow/strategic/generic) | Characterizes AI mentions |
| **RQ4** | Table 6: Interview sample and mechanism summary table | Mechanism evidence |
| **Supporting** | Figure 7: Source-specific robustness plots | Cross-source consistency |
| **Supporting** | Figure 8: Annotated break-analysis plot with candidate release windows | Temporal context |
| **Supporting** | Table: DiD regression coefficients for SWE-specific test | SWE-specificity |
| **All** | Table 2: Validation results for text measures | Measurement quality |
| **All** | Table 5: Sensitivity and robustness checks | Specification robustness |
| **All** | Figure: Specification curve for each primary finding | Robustness across 324 specifications |
| **All** | Table: Placebo test results | Falsification evidence |

---

## Stage 16: Statistical verification

This stage systematically challenges every finding from Stage 15. The goal is to show that results are not artifacts of sample composition, measurement error, random noise, or macro trends. A reviewer should be able to read this section and conclude that the findings survived serious attempts to break them.

Stage 16 runs AFTER Stage 15 produces initial results. We design the verification battery now but execute it only once we have findings to verify.

---

### 16a. Power analysis and sample size verification

If we lack statistical power to detect a plausible effect, a null result is uninformative and a significant result may be inflated (winner's curse). Confirm adequate power before interpreting results.

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

**Power table:**

| Test | Smaller group n | Minimum detectable effect | Adequate? |
|---|---|---|---|
| Junior share shift (chi-sq) | 3,466 | Cramer's V ~ 0.05 (3-5pp shift) | Yes |
| Description length (M-W) | 600 entry-level | Cohen's d ~ 0.14 | Yes |
| Skill prevalence per skill (z-test) | 600 entry-level | 4pp prevalence shift | Marginal — some rare skills may be underpowered |
| DiD interaction (sensitivity) | 3,466 SWE x 2024 | b3 ~ 0.05 SD | Yes |
| Embedding drift (bootstrap) | 600 entry-level | cos distance ~ 0.02 | Yes |
| Archetype Shift Index (bootstrap) | ~1,770 senior x 2024 | Ratio change ~ 0.10 | Yes |

**Decision rule:** If a test is underpowered (power < 0.60 for the hypothesized effect), do NOT run it as a primary test. Report it as exploratory or aggregate to a coarser level (e.g., combine rare skills into skill clusters).

---

### 16b. Within-period placebo tests

If our method detects a "change" between two halves of the same period (where no real change occurred), it is measuring noise. Every finding must pass this sanity check.

#### Placebo 1: Early vs. late Kaggle (within April 2024)

Split Kaggle data into two halves (before/after April 12). Run the same tests from Stage 15 on this split. All tests should produce null results.

**Sample size concern:** ~743 SWE postings in the early split is small. If tests are underpowered, compute the minimum detectable effect and report: "Our placebo test would have detected an effect of size X with 80% power. We observed null results, consistent with no within-period change, though small effects cannot be ruled out."

#### Placebo 2: Day-to-day variation in scraped data

Split 14 daily scrapes into two arbitrary weeks. Run the same comparison. We expect no significant differences — if we find them, daily variation is a concern and results need date-level controls.

**Stronger version:** Run the full test battery on every possible 7-day vs. 7-day split (C(14,7) = 3,432 splits). Compute the distribution of test statistics under these permutations. Our cross-period test statistic should be far outside this null distribution.

#### Placebo 3: Random subsample equivalence

Randomly split each period's SWE postings into two halves. Compare halves within each period. Neither comparison should yield significant results. Run 100 random splits and report the fraction producing p < 0.05. Under the null, this fraction should be ~5%.

---

### 16c. Control group verification

Our primary findings claim SWE roles changed between 2024 and 2026. If control occupations show the same changes, the finding is not SWE-specific.

#### Per-occupation controls

| Occupation | Kaggle n | Scraped n | Adequate for comparison? |
|---|---|---|---|
| Nursing | 6,082 | 5,360 | Yes — large sample |
| Accountant | 1,317 | 2,139 | Yes |
| Electrical eng | 377 | 1,435 | Marginal |
| Financial analyst | 372 | 1,417 | Marginal |
| Mechanical eng | 265 | 1,388 | Marginal |
| Civil eng | 157 | 547 | Underpowered — report but caveat |

Run every Stage 15 primary test on control occupations:

| Test | SWE result | Control expected | If control matches SWE |
|---|---|---|---|
| Junior share shift | Significant decline | No change | Macro trend, not AI-driven |
| Scope inflation | Significant increase | No change | Scraping artifact |
| Content convergence | High redefinition rate | Low/baseline rate | Classifier drift |
| AI keyword emergence | Significant increase | No change (or much smaller) | Expected differentiator |

#### AI-exposure gradient test

Order occupations by Felten et al. (2023) AI-exposure scores. Compute effect magnitude by occupation, plot against exposure score.

If correlation > 0.5 and significant, the effect scales with AI exposure — strong evidence of an AI-driven mechanism. This turns the SWE-specificity question into a continuous relationship.

#### Same-company cross-occupation test

For companies appearing in both SWE and non-SWE postings within both periods, compare whether the company's SWE postings changed differently than its non-SWE postings. This holds employer-level factors constant and isolates the occupation-level effect.

---

### 16d. Alternative explanation tests

Each test addresses a specific alternative explanation. If any alternative survives, acknowledge it as a limitation.

#### Alternative 1: "LinkedIn's algorithm/UI changed"

Compare the distribution of LinkedIn's native `job_level` labels between periods for both SWE and non-SWE postings. SWE-specific changes cannot be explained by platform-wide changes.

#### Alternative 2: "Composition effect (different companies)"

Within overlapping companies (appearing in both datasets), check whether the shift holds. Report: "X% of the observed shift is explained by composition changes; Y% remains after controlling for company overlap."

#### Alternative 3: "Scraping methodology artifact"

1. LinkedIn-only vs. Indeed-only within scraped data (same shift on both platforms?)
2. Geographic subset test: restrict both datasets to the same top-5 states
3. Company-matched test: only companies appearing in both datasets

#### Alternative 4: "Seasonality (April vs. March)"

Pull JOLTS monthly data for the information sector. Compare March vs. April historically (2019-2024). If the seasonal difference is typically <3%, seasonality cannot explain a 5+pp shift.

#### Alternative 5: "Ghost jobs driving results"

Run the full analysis excluding postings flagged as high ghost-job risk. If findings hold after exclusion, ghost jobs are not driving them. Report results both with and without ghost-flagged postings.

---

### 16e. Sensitivity analyses

Each sensitivity analysis asks: "If we made a different reasonable methodological choice, would we reach the same conclusion?"

#### Seniority classifier sensitivity

Run all primary tests under 3 classification schemes:

1. LLM-augmented classifier (default from Stage 10)
2. Rule-based only (from Stage 5)
3. Native LinkedIn labels where available, imputed only where missing

This sensitivity battery applies to the technical analysis corpus (`SWE` and, where relevant, `SWE-adjacent`). It is not a requirement that the default production run produce control-wide `seniority_llm`.

#### SWE definition sensitivity

1. Narrow: core SWE titles only (excluding data engineer, ML engineer)
2. Standard: canonical SWE_PATTERN (default)
3. Broad: include data scientist, QA engineer, product engineer

#### Deduplication sensitivity

1. No near-dedup (keep all postings with unique URLs)
2. Standard near-dedup (cosine >= 0.70)
3. Aggressive dedup (cosine >= 0.50, collapses multi-location)

#### Platform sensitivity

1. LinkedIn only (both periods on the same platform — most defensible comparison)
2. Full sample (LinkedIn + Indeed for 2026)

#### Company concentration sensitivity

1. Full sample
2. Capped at 10 postings per company
3. Excluding top-5 companies from each dataset

#### Geographic sensitivity

1. Full sample
2. Top-5 metros only (matched across datasets)
3. Excluding remote postings

#### Metro-balanced subsamples

Subsample scraped data to match Kaggle geographic distribution. Rerun primary tests.

#### Canonical postings vs. daily observations

Run analyses on both `unified.parquet` (canonical postings, deduplicated) and `unified_observations.parquet` (daily observations, includes reposts). If findings differ, investigate whether repost dynamics are driving results.

#### Consolidated sensitivity table

| Finding | Default | Alt seniority | Alt SWE def | LinkedIn-only | Cap 10/co | Top-5 metros | Canonical vs. obs | Verdict |
|---|---|---|---|---|---|---|---|---|
| Junior share decline | p=X, d=Y | ... | ... | ... | ... | ... | ... | Robust/Suggestive/Fragile |
| Scope inflation | ... | ... | ... | ... | ... | ... | ... | ... |
| Senior archetype shift | ... | ... | ... | ... | ... | ... | ... | ... |
| Skill migration (top skill) | ... | ... | ... | ... | ... | ... | ... | ... |
| Divergence index | ... | ... | ... | ... | ... | ... | ... | ... |

**Verdict rules:**

- **Robust:** Significant (p < 0.05) and same direction in >= 5 of 7 sensitivity variants
- **Suggestive:** Significant and same direction in 3-4 of 7 variants
- **Fragile:** Significant in < 3 variants — not reported as a finding

---

### 16f. Effect size calibration

A statistically significant result with a tiny effect size is practically meaningless. Calibrate what effect sizes mean in context.

#### External benchmarks

| Metric | Our observed change | Context benchmark | Interpretation |
|---|---|---|---|
| Junior share change | X pp | Hershbein & Kahn (2018): junior share dropped ~5pp during Great Recession | If similar magnitude, comparable to a recession-level shock |
| Description length change | X words | Deming & Kahn (2018): skill requirements increased ~20% 2007-2017 | Decade-scale baseline for scope inflation |
| AI keyword emergence | X pp | Acemoglu et al. (2022): AI-exposed occupations saw 15-20% task reallocation | Calibrates meaningful AI adoption |
| Embedding drift (cosine) | X | Need to establish: what is the typical within-period drift? | If cross-period >> within-period, the change is real |

#### Practical significance thresholds (set before looking at results)

- **Junior share change:** >= 3 percentage points
- **Skill prevalence change:** >= 5 percentage points for any individual skill
- **Description length change:** >= 15% increase in requirements section
- **Redefinition rate:** >= 10 percentage points above baseline misclassification rate
- **DiD interaction (sensitivity):** >= 0.10 SD
- **Archetype Shift Index change:** >= 20% relative increase
- **Divergence index change:** >= 10 percentage points widening

Findings that cross both statistical and practical significance thresholds are "strong evidence." Findings that are statistically significant but below practical thresholds are "detectable but modest."

---

### 16g. Reproducibility protocol

1. **Random seeds:** Set `np.random.seed(42)` and `torch.manual_seed(42)` at the start of every analysis script. Report the seed.
2. **Model versioning:** Pin all HuggingFace model versions by commit hash (not just model name).
3. **Data versioning:** Hash the unified.parquet file (SHA-256) and report in methodology. Any re-run must produce the same hash after preprocessing.
4. **Pipeline code:** All preprocessing and analysis code in version-controlled scripts. No manual steps between preprocessing and results.
5. **Intermediate outputs:** Cache and version all embeddings, topic models, and classifier predictions. A reviewer should be able to start from cached embeddings and reproduce all results without re-running the embedding step.

---

### Verification outputs

| Output | Type | Paper section |
|---|---|---|
| Power analysis table | Table | Methodology |
| Within-period placebo results (3 tests) | Table | Placebo tests |
| Control group results (per occupation + pooled) | Table + figure | Control analysis |
| AI-exposure gradient plot | Figure | Control analysis (if gradient confirmed) |
| Alternative explanation tests (5 tests) | Table | Discussion or robustness |
| Sensitivity analysis matrix | Table (consolidated) | Sensitivity |
| Specification curve plots (per finding) | Figures | Sensitivity or appendix |
| Effect size calibration table | Table | Alongside main results |
| Reproducibility metadata | Appendix | Appendix |

---

## Implementation order

### Phase 5: Formal Analysis (Stage 15)

```
33. RQ1 seniority shift + content convergence + scope inflation (15a Tests 1-3)
34. RQ1 senior archetype shift + Fightin' Words + BERTopic (15a Tests 4-6)
35. RQ1 controlled regressions + heterogeneity analysis (15a Tests 7-8)
36. RQ2 skill prevalence shifts + Fightin' Words + co-occurrence networks (15b)
37. RQ3 divergence index construction + cross-occupation comparison (15c)
38. RQ4 interview analysis with reflexive thematic analysis (15d)
39. Supporting: break analysis + DiD sensitivity + embedding drift (15e)
40. Specification curves + multiple testing corrections + bootstrap CIs (15f)
```

### Phase 6: Statistical Verification (after Stage 15 produces results)

```
41. Power analysis for every primary test (16a)
42. Within-period placebos: early/late Kaggle, week 1/2 scraped, random splits (16b)
43. Control group verification: per-occupation + pooled + AI-exposure gradient (16c)
44. Alternative explanation tests: platform, composition, scraping, seasonality, ghost (16d)
45. Sensitivity matrix: seniority x SWE def x dedup x platform x company x geography x canonical-vs-obs (16e)
46. Effect size calibration against external benchmarks (16f)
47. Reproducibility protocol: seeds, model versions, data hashes (16g)
```
