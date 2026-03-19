# Statistical Methods Reference for the SWE Labor Study

Date: 2026-03-18
Source: Consolidated from 6 research documents. This is the standalone reference --
all code examples and sources are included below.

---

## How to use this document

This reference is organized by **what you're trying to do** in the study, not by method family. Each section maps to one or more research questions, lists the applicable methods from simplest to most sophisticated, and flags the key decision points.

---

## 1. Comparing language across groups (RQ1, RQ2, RQ6)

*"What words/skills distinguish junior from senior postings? How has the language of junior roles changed between 2024 and 2026?"*

### Fightin' Words (log-odds-ratio with Dirichlet prior)

The right tool for pairwise corpus comparison. Produces both effect size (log-odds-ratio) and significance (z-score) while regularizing rare words through an informative prior. Handles corpus size imbalances gracefully.

**Use for:**
- Junior vs. senior posting language
- 2024 vs. 2026 posting language within seniority levels
- SWE vs. control occupation language

**Key advantage over TF-IDF:** TF-IDF zeros out words appearing in both corpora, even if used at very different rates. Fightin' Words catches rate differences.

**Python:** `pip install fightin-words` (sklearn-compliant) or ConvoKit's `FightingWords` class.

**Watch out for:** Bag-of-words assumption (no context); must pre-define groups; multiple testing with 10K+ vocabulary words.

*See section A2 below for full details and code.*

### Jensen-Shannon Divergence (JSD)

Measures how different two probability distributions are (e.g., skill distributions across two time periods). Symmetric, bounded [0, 1], and defined even when distributions have zero-count entries. Use as a single-number summary of "how different are these two corpora?"

**Python:** `scipy.spatial.distance.jensenshannon(p, q)`

### Kolmogorov-Smirnov Test

Nonparametric test for whether two samples come from the same distribution. Use for continuous variables (description length, salary, years of experience required).

**Python:** `scipy.stats.ks_2samp(sample1, sample2)`

---

## 2. Discovering latent structure in postings (RQ2, exploratory)

*"What skill clusters exist? What topics are emerging that we didn't know to look for?"*

### BERTopic (recommended starting point)

Modular pipeline: sentence embedding -> UMAP -> HDBSCAN -> c-TF-IDF. Automatically determines number of topics, captures semantic meaning, supports temporal analysis.

**Key parameters to tune:**
- Embedding model: `all-MiniLM-L6-v2` (fast, 384d) or `all-mpnet-base-v2` (better quality, 768d)
- `min_topic_size`: 50 for broad topics, 10-20 for granular (our corpus ~200K docs, start at 50)
- UMAP `n_neighbors`: 15-30 for job postings

**Critical preprocessing:** Remove EEO statements, benefits sections, company boilerplate. This is the single highest-impact step.

**Limitation:** Hard clustering (one topic per document). Job postings often span multiple skill areas. For mixed-membership, consider LDA.

### BERTrend (for emerging topic detection)

Runs BERTopic independently per time slice, then merges topics across windows using cosine similarity (threshold 0.7). Can detect genuinely new topics emerging -- unlike standard BERTopic which assumes a fixed topic set.

**Critical for our study:** The emergence of AI-native skills in 2025-2026 violates BERTopic's fixed-topic assumption. BERTrend handles this.

**Compute:** Runs on CPU for our corpus size (~200K docs). Embedding step ~30-60 min on 20 cores with small model.

### LDA (when mixed membership matters)

Outperformed BERTopic on wage prediction in a 1.16M posting study because mixed membership better captures how postings span multiple skill domains. Consider for analyses where a posting being "40% cloud + 30% data + 30% leadership" matters.

**Python:** `gensim.models.LdaModel` or `sklearn.decomposition.LatentDirichletAllocation`

### NMF (deterministic baseline)

Same input, same output every time. Clear topic separation. Good baseline to run alongside BERTopic for robustness.

**Python:** `sklearn.decomposition.NMF`

*See section A1 below for full details and code.*

---

## 3. Detecting structural breaks (RQ3)

*"Did the junior SWE market experience a discrete regime change in late 2025?"*

### Bai-Perron (gold standard for endogenous breakpoints)

Finds multiple structural breaks at unknown dates in time series. The data tells you when things changed, rather than you specifying dates. Uses dynamic programming (O(T^2)) for efficient global optimization.

**Testing workflow:**
1. UDmax/WDmax: Any breaks at all?
2. Sequential sup F(l+1|l): How many breaks?
3. Report confidence intervals for break dates

**Our constraint:** ~3 months post-break data (if break is Dec 2025). Sufficient for level shifts, limited power for slope changes.

**Python:** `ruptures` library or `statsmodels` (limited Bai-Perron support). R's `strucchange` package is more complete.

### Interrupted Time Series (ITS) / Segmented Regression

Models a time series as having different levels and slopes before and after a known intervention date:

```
y_t = B0 + B1(time) + B2(post) + B3(time x post) + e_t
```

- B2: immediate level change at the intervention
- B3: change in slope after the intervention

**Use for:** Testing whether December 2025 (agent deployment) produced a shift in junior posting share, skill breadth, or AI-mention rates.

### Bayesian Online Change Point Detection (BOCPD)

Multivariate change point detection -- pools signals across multiple feature streams simultaneously. Higher power than univariate methods because it detects coordinated shifts.

**Use for:** Running on multiple indicators simultaneously (posting volume + skill breadth + embedding drift + AI-keyword prevalence). If multiple streams flag the same quarter, that's strong evidence.

**Python:** `bayesian_changepoint_detection` package or `ruptures` with `Binseg`/`Pelt` algorithms.

### Chow Test (confirmatory)

Tests for a structural break at a known date. Use as a confirmatory test after Bai-Perron identifies a candidate break. Also useful for placebo tests at arbitrary dates.

**Sensitivity check:** Run BOCPD on rolling windows excluding the most recent 1, 2, 3 months to test whether detected breaks are stable or endpoint artifacts.

*See section B1 below for full details and code.*

---

## 4. Establishing causation (RQ3, RQ4)

*"Is the SWE shift caused by AI agents, or is it a broader macro trend?"*

### Difference-in-Differences (DiD)

The workhorse causal inference method. Compare SWE postings (treatment) against non-AI-exposed occupations (control) before and after the agent deployment date. The key estimand is the interaction term: SWE x Post-Agent.

**Critical assumption:** Parallel trends pre-treatment. Must show that SWE and control occupations were on similar trajectories before the break.

**Control selection:** Use Felten et al. (2023) or Eloundou et al. (2023) AI exposure scores to select low-AI-exposure occupations (civil engineering, nursing, mechanical engineering).

### Synthetic Control Method

Instead of hand-picking control occupations, construct a weighted combination of all non-AI-exposed occupations that best matches pre-treatment SWE trends. More defensible than arbitrary control selection.

**Robustness:** Run placebo synthetic controls for each donor occupation. If many placebos show gaps as large as SWE's, the finding is not significant.

**Python:** `SparseSC` or `SyntheticControlMethods`

### Event Study Design

Estimate period-by-period treatment effects around the intervention date. The pre-treatment coefficients serve as a visual parallel-trends test. Should show null effects before treatment and growing effects after.

**Implementation:** Interact time dummies with treatment indicator; plot coefficient estimates with confidence intervals.

*See section B1 below for full details and code.*

---

## 5. Classifying and measuring text features (RQ1, RQ2, RQ5, RQ6)

*"Is this posting junior or senior? What skills does it require? Is the seniority classifier accurate?"*

### Sentence Transformers (SBERT) -- embedding everything

Map job postings to dense vectors where semantic similarity = vector proximity. Foundation for clustering, drift measurement, and similarity analysis.

**Model recommendations:**
- `all-MiniLM-L6-v2`: Fast, 384d, good for CPU. Use for initial exploration.
- `all-mpnet-base-v2`: Better quality, 768d. Use for final analysis.
- **Domain-adapted models** (JobBERT, SkillBERT): Consistently outperform generic models on job posting tasks. Fine-tuning on job posting text is the single biggest accuracy improvement.

**Embedding drift analysis:** Compute average embedding of "junior SWE" postings per quarter. Track how this centroid moves toward the "senior SWE" centroid over time. This directly measures content convergence (RQ1).

### SetFit (few-shot classification)

Train a classifier with only 8-16 labeled examples per class. Uses contrastive learning on sentence embeddings. No GPU required.

**Use for:** Seniority classification, SWE/non-SWE detection -- especially if we create a gold-standard annotation set (300-500 labels, as recommended in the validation research).

**Python:** `pip install setfit`

### Zero-Shot Classification

Classify text without any training data using NLI models. Use `facebook/bart-large-mnli` or `cross-encoder/nli-deberta-v3-base`. Good for rapid prototyping and validation against keyword classifiers.

**Python:** `transformers` pipeline with `zero-shot-classification`

### Skill Extraction (NER-based)

**SkillSpan** (Zhang et al. 2022, NAACL): Domain-specific NER for hard and soft skills. 14.5K annotated sentences, F1 = 56-64%.

**ESCO/O\*NET mapping:** Map extracted skills to standard taxonomies for cross-study comparability.

**Practical approach:** Use LLM-assisted extraction (GPT-4o-mini or Haiku) for structured skill lists at ~$0.001/posting, then map to taxonomy.

*See section B2 below for full details and code.*

---

## 6. LLM-assisted qualitative analysis (RQ2, RQ5, RQ6)

*"What implicit patterns exist in how skill requirements are described? What does 'AI literacy' actually mean in these postings?"*

### LLM Thematic Analysis (Braun & Clarke adapted)

Use LLMs for scalable qualitative coding. The emerging best practice is human-LLM collaboration: LLMs handle initial coding, humans retain interpretive authority.

**Key findings from the literature:**
- GPT-4 with chain-of-thought prompting: mean kappa = 0.68 (substantial agreement with human coders)
- Per-code prompting outperforms full-codebook prompting
- Cost: ~$60-120 for 100K postings with 10 codes via Batch API

**Validation requirement:** Must compute Cohen's kappa against human annotations. Target kappa > 0.6 per code before scaling.

**Critical warning (Ashwin et al. 2025):** LLM coding errors are NOT random -- they correlate with characteristics of the text. Can produce completely incorrect conclusions. Always validate on a gold standard.

### Two-stage pipeline for job postings

1. **Stage 1 (cheap model):** GPT-4o-mini / Claude Haiku for structured extraction (skills, requirements sections, seniority signals). Classification task -- cheap models perform well.
2. **Stage 2 (better model):** GPT-4o / Claude Sonnet for thematic analysis on cleaned, structured data. Discovery task -- needs stronger model.

*See section A3 below for full details and code.*

---

## 7. Regression and econometric modeling (RQ1-RQ4)

*"After controlling for industry, firm size, and geography, did junior skill breadth increase?"*

### OLS with text-derived features

Standard approach: regress outcomes (wage, application count) on text-extracted features (skill counts, AI-keyword frequencies, topic shares). Must use robust standard errors (HC3 recommended) and cluster by firm or metro area.

**Critical issue (Ash & Hansen 2023):** Text quantification and econometric models are usually treated separately, which creates inference problems. Downstream regression ignores upstream measurement uncertainty. Valid inference requires bias correction or joint estimation.

### Panel data methods (fixed effects)

If tracking firms or occupations across time, use firm x occupation fixed effects to absorb unobserved heterogeneity. This is what Acemoglu et al. (2022) and Chen & Stratton (2026) do.

### Index construction

**Skill Breadth Index:** Count of distinct skill categories per posting. Track over time within seniority levels. Rising SBI in junior postings = scope inflation.

**Senior Keyword Infiltration Index:** Fraction of traditionally-senior keywords appearing in junior postings. Direct measure of competency migration (RQ2).

**Archetype Shift Index:** Ratio of people-management keywords to AI-orchestration keywords in senior postings. Tracks the manager->orchestrator shift (RQ6).

### Survival / duration analysis

**Time-to-fill** as a labor market signal. Kaplan-Meier curves and Cox proportional hazards models. If junior postings are taking longer to fill, that suggests either fewer qualified candidates or mismatched requirements.

### Network methods

**Skill co-occurrence networks:** Nodes = skills, edges = co-occurrence in the same posting. Track how network structure changes over time (new connections appearing = skill migration). Centrality measures identify which skills anchor others.

*See section B3 below for full details and code.*

---

## 8. Multiple testing and robustness (all RQs)

### Specification curve analysis

Run core analysis under all defensible specifications (alternative occupation definitions, dedup thresholds, seniority classifiers, geographic subsets). Plot the distribution of estimates.

**Python:** `pip install specification-curve`

### Placebo tests

1. Run cross-period comparison on **control occupations** -- should show no "structural change"
2. Run on **placebo time split** within Kaggle data -- should show no "break"
3. Pre-trend analysis within Kaggle months

### Multiple testing corrections

With thousands of skill terms, chi-squared tests, etc., many will be "significant" by chance. Apply:
- **Benjamini-Hochberg FDR** (preferred): Controls false discovery rate at 5%. Less conservative than Bonferroni.
- **Bonferroni** (conservative): Divides alpha by number of tests. Use when false positives are very costly.

### Bootstrap confidence intervals

Resample with replacement 1000+ times. Report bootstrap CIs for all key estimates. Especially important for text-derived metrics where analytic standard errors are unreliable.

**Python:** `scipy.stats.bootstrap` or manual loop with `joblib.Parallel`.

*See section B3 below for full details and code.*

---

## Method-to-RQ mapping

| RQ | Primary methods | Secondary methods |
|----|----------------|-------------------|
| **RQ1** (disappearing vs. redefined?) | Embedding drift, seniority classifier, Fightin' Words (2024 vs 2026 junior postings) | SetFit, JSD on skill distributions |
| **RQ2** (which skills migrated?) | Skill extraction + prevalence curves, BERTopic, skill co-occurrence networks | Fightin' Words (junior vs senior), LLM thematic analysis |
| **RQ3** (structural break?) | Bai-Perron, ITS, BOCPD (multivariate) | Chow test (confirmatory), event study |
| **RQ4** (SWE-specific?) | DiD, synthetic control | Placebo tests on control occupations |
| **RQ5** (training implications) | Qualitative synthesis from RQ1-4 outputs | Cross-profession comparison (qualitative) |
| **RQ6** (senior role shift?) | Fightin' Words (management vs orchestration keywords), Archetype Shift Index | BERTopic on senior postings, LLM coding |
| **All** (robustness) | Specification curves, placebo tests, bootstrap CIs, FDR correction | Multiple definitions, subsample analyses |

---

# FULL METHOD DETAILS

# Part A: Text Analysis Methods

## A1. Topic Modeling for Labor Market Analysis

### 1. What is topic modeling?

Topic modeling is a family of unsupervised machine learning techniques that discover latent thematic structures ("topics") in a collection of documents. Each topic is represented as a cluster of frequently co-occurring words, and each document is associated with one or more topics.

In the context of job postings, topic modeling can automatically surface the skill clusters, role types, and competency patterns embedded in millions of unstructured job descriptions -- revealing labor market structure and trends that manual taxonomy approaches cannot scale to capture.

**The relationship between topic modeling and keyword analysis:**

| | Keyword Analysis | Topic Modeling |
|---|---|---|
| **Finds** | Things you already know to look for | Things you didn't know existed |
| **Misses** | Emergent skills, semantic variation | Specific frequencies of known skills |
| **Precision** | High (you defined the terms) | Low (topics are fuzzy clusters) |
| **Recall** | Low (limited to your dictionary) | High (covers the whole corpus) |
| **Best for** | Hypothesis testing (RQ2, RQ6) | Hypothesis generation (discovering what else is changing) |

The ideal workflow: run topic modeling first to discover the space, then use keyword analysis to precisely measure the patterns you find.

---

### 2. Technique quick reference

#### LDA (Latent Dirichlet Allocation)

Probabilistic model that treats every document as a mixture of topics. Produces mixed-membership (a posting can be 40% cloud + 30% data + 30% leadership). Bag-of-words, requires specifying K topics in advance, needs heavy preprocessing. Use when mixed membership matters or for comparability with prior literature. **Python:** `gensim.models.LdaModel`, `sklearn.decomposition.LatentDirichletAllocation`

#### NMF (Non-negative Matrix Factorization)

Factors a document-term matrix into document-topic and topic-word matrices. Deterministic (same input = same output), clear topic separation, fast. Bag-of-words, requires specifying K. Good baseline to run alongside BERTopic for reproducibility. **Python:** `sklearn.decomposition.NMF`

#### Top2Vec

Similar to BERTopic but defines topics by proximity in embedding space (nearest word vectors to cluster centroids) rather than c-TF-IDF. Auto-detects topic count, no preprocessing needed. Tends to produce excessive overlapping topics; less actively maintained than BERTopic. **Python:** `top2vec`

#### FASTopic

Uses transport plans and the Sinkhorn algorithm for fast topic modeling with superior coherence scores. No per-document topic assignment -- analyzes the corpus holistically. Use when you need speed and don't need document-level assignments.

#### Other approaches

- **BunkaTopics:** SBERT + UMAP + K-means with collaborative visualization. Good for exploratory analysis.
- **LLM-based topic modeling:** Use BERTopic for clustering, then LLMs for human-readable topic labels. A 2025 study found Phi achieved highest topic diversity (0.717) while Qwen achieved highest relevance (0.610).
- **Seeded topic modeling:** Provide seed words to guide topic discovery. Useful when you have prior expectations about what topics should exist.

---

### 3. BERTopic in detail

**How it works:**
1. **Embed**: Convert each document into a dense vector using a pre-trained sentence transformer.
2. **Reduce**: Use UMAP to compress the high-dimensional embeddings while preserving local structure.
3. **Cluster**: Apply HDBSCAN to find dense regions of semantically similar documents. Documents in sparse regions become outliers (topic -1).
4. **Represent**: Extract topic labels using class-based TF-IDF (c-TF-IDF) -- identifying words distinctive to each cluster compared to all others.

**Strengths:**
- Captures semantic meaning -- understands that "ML engineer" and "machine learning developer" are related
- Automatically determines number of topics
- Minimal preprocessing required
- Modular -- can swap embedding models, clustering algorithms, or representation methods
- Built-in temporal analysis, guided topics, hierarchical reduction, LLM labeling

**Weaknesses:**
- Hard clustering: one topic per document (no mixed membership)
- Can produce many small, noisy topics that need manual review/merging
- Outliers can be 10-30% of corpus
- Computationally expensive (embedding step)
- Sensitive to UMAP/HDBSCAN hyperparameters

**Key parameters:**

| Parameter | Where | Guidance |
|-----------|-------|---------|
| Embedding model | SentenceTransformer | `all-MiniLM-L6-v2` (fast) or `all-mpnet-base-v2` (quality) or domain-adapted (JobBERT) |
| `min_topic_size` | BERTopic | 50 for ~200K docs (broad topics), 10-20 for granular |
| `n_neighbors` | UMAP | 15-30 for job postings. Low (5-10) = fine clusters, high (30-50) = broad structure |
| `n_components` | UMAP | 5-10 for clustering (not 2) |
| `min_dist` | UMAP | 0.0 for clustering |
| `min_cluster_size` | HDBSCAN | Same as `min_topic_size`. Start at 50, adjust by corpus size |

**Temporal analysis parameters:**
- `nr_bins`: Group timestamps into N bins. Recommended when unique timestamps > 50.
- `global_tuning` (default True): Anchor topic representations to global average.
- `evolutionary_tuning` (default True): Smooth transitions between adjacent periods.

**Limitation of `topics_over_time()`:** Topics are defined globally (same set across all time), only keywords change. Cannot capture genuinely new topics emerging. Use BERTrend for that.

---

### 4. BERTrend in detail

**Framework for online/streaming trend detection:**

1. Slice corpus into time windows (daily, weekly, monthly)
2. Run BERTopic independently on each time slice
3. Merge topics across consecutive windows using cosine similarity (threshold 0.7)
4. Track topic popularity with exponential decay
5. Classify topics as noise, weak signals, or strong signals based on dynamic percentile thresholds

**Implementation parameters:**
- Embedding model: `all-mpnet-base-v2`
- UMAP: 5 components, 15 neighbors
- HDBSCAN: min cluster size 2
- Merge threshold: 0.7 cosine similarity
- Decay factor: 0.01

**Strengths:** Detects genuinely new emerging topics. Designed for monitoring evolving corpora.
**Weaknesses:** More complex to implement. Open-sourced at `rte-france/BERTrend`.

**Recommendations for tracking skill migration:**
1. Start with BERTopic `topics_over_time()` for initial exploration
2. If you need to detect genuinely new skill topics, use BERTrend's approach
3. Domain-adapt your embedding model on job posting text before running temporal analysis

---

### 5. Technique comparison

| Dimension | LDA | NMF | BERTopic | Top2Vec |
|-----------|-----|-----|----------|---------|
| **Semantic understanding** | None (bag-of-words) | None (bag-of-words) | Strong (transformers) | Strong (embeddings) |
| **Mixed membership** | Yes (soft) | Yes (soft) | No (hard) | No (hard) |
| **Number of topics** | Must specify K | Must specify K | Automatic | Automatic |
| **Preprocessing needed** | Heavy | Heavy | Minimal | Minimal |
| **Deterministic** | No | Yes | No | No |
| **Short text performance** | Poor | Decent | Good | Decent |
| **Compute requirements** | CPU-friendly | CPU-friendly | GPU preferred | GPU preferred |
| **Dynamic/temporal** | Via DTM extension | Limited | Built-in | Not built-in |
| **Outlier handling** | None (all assigned) | None (all assigned) | Outlier class (-1) | Outlier class |

**When to use which:**
- **LDA**: Mixed membership matters, limited compute, or comparability with prior literature.
- **NMF**: Deterministic results, clear separation, shorter texts. Good baseline.
- **BERTopic**: Semantic understanding matters, unknown topic count, temporal analysis. Best default for new projects.
- **Top2Vec**: Simple one-line API. BERTopic generally preferred.

---

### 6. Evaluation metrics

- **C_v coherence**: Combines word co-occurrence and sliding window. Higher is better. Typical range: 0.3-0.7.
- **C_NPMI**: Measures whether word pairs co-occur more than expected by chance. Range: -1 to 1, 0.05-0.2 typical.
- **Topic diversity**: Fraction of unique words across all topics' top-N keywords.
- **Perplexity** (LDA only): How well the model predicts held-out documents. Lower is better, but weakly correlated with human-judged quality.

No automated metric reliably tells you whether topics are useful for your research question. Human evaluation -- reading the top documents per topic -- remains essential.

---

### 7. Application to job postings -- literature and findings

**Key studies:**
- **World Bank (2024)**: BERTopic on Latin American job postings. Domain adaptation of BERT on job posting text was critical.
- **UK Job Board Study (1.16M postings)**: LDA explained 48.3% of wage variation, BERTopic 32.6%. LDA won because mixed membership better captures multi-skill postings.
- **UK AI/Green Job Vacancies (2025)**: ~11M UK vacancies 2018-2024. Found shift toward skill-based hiring, especially for AI roles.

**What worked in practice:**
1. Domain-adapted embeddings make a significant difference.
2. LDA can outperform BERTopic on large, well-structured datasets where mixed membership matters.
3. Preprocessing matters more for job postings than typical text -- boilerplate dominates if not removed.
4. Hierarchical topic reduction is almost always necessary.

---

### 8. Preprocessing for job postings

**For LDA/NMF (heavy preprocessing):**
- Remove HTML tags, URLs, email addresses
- Remove boilerplate (EEO statements, benefits, company descriptions)
- Lowercase, remove stopwords (standard + domain: "experience", "required", "skills", "ability", "must")
- Lemmatize
- Remove very rare (< 5 docs) and very common (> 50% docs) words

**For BERTopic/Top2Vec (minimal preprocessing):**
- Remove HTML tags, URLs, email addresses
- Remove boilerplate (still critical!)
- Optionally segment long postings into paragraphs (embedding models truncate at 256-512 tokens)

**Job-posting-specific tips:**
- Boilerplate removal is the single highest-impact step.
- Consider extracting just the "requirements" or "qualifications" sections.
- Treat job titles as separate metadata rather than mixing into document text.
- Remove salary/location from text but preserve as metadata.

---

### 9. Common pitfalls

1. **Skipping boilerplate removal**: You get topics like "equal opportunity employer disability veteran" instead of skill clusters.
2. **Wrong number of topics**: Too few = distinctions lost. Too many = fragmented. Use coherence scores for LDA/NMF; `reduce_topics()` for BERTopic.
3. **Not evaluating qualitatively**: Always inspect top documents per topic, not just top words.
4. **Treating topics as ground truth**: Topics are one of many possible decompositions. Use as discovery, validate with targeted methods.
5. **Ignoring outliers**: BERTopic assigns 10-30% to topic -1. Use `reduce_outliers()` or account for them.
6. **Using generic embeddings**: Fine-tune or use domain-adapted models (JobBERT, JobSpanBERT).
7. **Not accounting for corpus composition changes**: Normalize by sector/occupation when possible.
8. **Conflating frequency with importance**: "Communication skills" appears everywhere but tells you nothing.
9. **Non-reproducibility**: Set random seeds everywhere. Run multiple times. NMF is deterministic.
10. **Token limits**: Most sentence transformers truncate at 256-512 tokens. Segment long postings or use the most informative section.

---

### 10. Key sources

- Egger & Yu (2022). "A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic" -- PMC
- MDPI Sustainability (2025). "Towards a Sustainable Workforce in Big Data Analytics" -- Neural topic modeling on job postings
- World Bank (2024). "Understanding Labor Market Demand" -- BERTopic on Latin American postings
- ScienceDirect (2025). "Skills or degree? The rise of skill-based hiring" -- 11M UK job vacancies
- Zhang et al. (2022). "SkillSpan: Hard and Soft Skill Extraction from English Job Postings" -- NAACL
- BERTrend (2024). "Neural Topic Modeling for Emerging Trends Detection" -- arXiv 2411.05930
- BERTopic documentation: topics_over_time feature

---

## A2. Fightin' Words: Comparative Corpus Analysis

### 1. What it is

**Fightin' Words** is a statistical method for identifying which words (or n-grams) are distinctively associated with one group of texts versus another. Introduced by Monroe, Colaresi, and Quinn (2008) in *Political Analysis*.

Given two corpora (e.g., junior vs. senior postings, 2024 vs. 2026 postings), it identifies which words each group uses distinctively more than the other, with both effect size and significance while regularizing rare words.

### 2. The z-score formula

For each word w, the Fightin' Words z-score is:

```
z_w = delta_hat_w / sqrt(sigma^2_w)
```

Where the log-odds-ratio estimate is:

```
delta_hat_w = log((y_w^i + alpha_w) / (n^i + alpha_0 - y_w^i - alpha_w))
            - log((y_w^j + alpha_w) / (n^j + alpha_0 - y_w^j - alpha_w))
```

And the variance is:

```
sigma^2_w = 1/(y_w^i + alpha_w) + 1/(y_w^j + alpha_w)
```

- `y_w^i`, `y_w^j` = count of word w in corpus i, j
- `n^i`, `n^j` = total word count in corpus i, j
- `alpha_w` = Dirichlet prior, set proportional to word frequency in the combined background corpus
- `alpha_0` = sum of all alpha_w

**What the z-score means:** Values outside [-1.96, 1.96] indicate 95% significance. Positive = more distinctive of corpus i; negative = corpus j. The informative prior shrinks rare words toward the background rate, preventing them from producing spurious extreme scores.

### 3. How it compares to simpler approaches

| Method | Handles size imbalance? | Regularizes rare words? | Effect size? | Significance? |
|--------|------------------------|------------------------|-------------|--------------|
| Raw frequency | No | No | Sort of | No |
| TF-IDF | Partially | No | Yes | No |
| Chi-squared | Yes | No | No | Yes |
| Log-likelihood ratio | Yes | No | No | Yes |
| Simple log-odds | Yes | No (or ad-hoc) | Yes | No |
| **Fightin' Words** | **Yes** | **Yes (principled)** | **Yes** | **Yes** |

**vs. TF-IDF:** TF-IDF zeros out any word appearing in all documents. A word used by both groups at very different rates gets zero weight. Fightin' Words catches these rate differences.

**vs. Chi-squared:** Chi-squared gives equal weight to rare and common words. A word appearing once in one corpus and zero in the other gets flagged as significant. The Dirichlet prior prevents this.

### 4. Applications in labor market analysis

- **Comparing occupations**: "What words distinguish data engineer postings from data scientist postings?"
- **Temporal comparison**: "How has junior SWE language changed from 2024 to 2026?"
- **Seniority comparison**: "What language distinguishes junior from senior postings?"
- Handles corpus size imbalances gracefully
- Skill terms can be rare -- the prior prevents spurious results while surfacing genuinely distinctive ones

### 5. Python implementations

#### Option 1: fightin-words (PyPI)
Scikit-learn compliant. Returns (word, z-score) tuples.

#### Option 2: ConvoKit (Cornell)
Built-in visualization with `plot_fighting_words()`.

#### Option 3: kornosk/log-odds-ratio (GitHub)

#### Option 4: Scattertext

#### R: tidylo
Gold standard for tidy R workflows. Output `log_odds_weighted` column is both a z-score and weighted log-odds-ratio.

### 6. Limitations and pitfalls

1. **Bag-of-words**: No context. "Bank" in finance vs. geography is the same word. Mitigated by using n-grams.
2. **Requires pre-defined groups**: Cannot discover groups -- that's what topic modeling does.
3. **Pairwise only**: Multi-group requires pairwise or one-vs-rest comparisons.
4. **No semantics**: Finds *which* words differ, not *why*. Human interpretation required.
5. **Prior size is consequential**: Too large shrinks genuine differences; too small brings back rare-word problems. Empirical Bayes default is sensible.
6. **Stopword decisions matter**: Function words can be interesting or can swamp content words.
7. **Multiple testing**: With 10K+ words, many cross z > 1.96 by chance. Focus on magnitude of z-scores, not strict cutoffs.

### 7. Relationship to topic modeling

| | Fightin' Words | Topic Modeling |
|---|---|---|
| **Question** | "How do group A and group B differ?" | "What latent themes exist?" |
| **Input** | Two pre-defined groups | A single corpus |
| **Output** | Ranked distinctive words per group | Clusters of co-occurring words |
| **Supervision** | Supervised (groups labeled) | Unsupervised |

Best used together: topic modeling first to discover themes, then Fightin' Words to compare how groups differ within or across themes.

### 8. Key sources

- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' Words. *Political Analysis*, 16(4), 372-403.
- ConvoKit Fighting Words: https://convokit.cornell.edu/documentation/fightingwords.html
- kornosk/log-odds-ratio: https://github.com/kornosk/log-odds-ratio
- tidylo package: https://juliasilge.github.io/tidylo/
- fightin-words PyPI: https://pypi.org/project/fightin-words/

---

## A3. LLM-Assisted Thematic Analysis

### 1. What inductive thematic analysis is

Thematic analysis (TA) identifies patterns (themes) within qualitative data. The Braun & Clarke (2006) six phases: familiarization, generating codes, searching for themes, reviewing themes, defining themes, writing up.

- **Inductive**: Codes and themes emerge from the data. Bottom-up.
- **Deductive**: Codes from a pre-specified framework. Top-down.

Traditional TA is labor-intensive. Scaling to millions of job postings is impossible with manual methods alone.

### 2. How LLMs are being used

**What the research shows:**
- GPT-4 with chain-of-thought: mean Cohen's kappa = 0.68 (substantial agreement with human coders)
- Per-code prompting outperforms full-codebook prompting
- LLMs reduce coding time from weeks to hours
- LLMs are better at concrete, descriptive themes than subtle, interpretive ones

**The consensus:** LLMs complement human analysts rather than replace them. LLMs handle scalable coding; humans retain interpretive authority.

### 3. Methodologies

#### Chain-of-thought prompting
Four-component prompt: (1) role assignment, (2) code definition, (3) justification request, (4) structured decision output. CoT improved GPT-4 agreement from 0.59 to 0.68.

#### Per-code vs. full-codebook
Per-code (one prompt per code) consistently outperforms full-codebook. More expensive but more accurate.

#### Few-shot
Largest gain is zero-shot to one-shot. Codebook-centered designs outperform example-centered. GPT-4 achieved kappa 0.738 on requirements classification with full context.

### 4. Validation

**Cohen's kappa interpretation:** <0.20 poor; 0.21-0.40 fair; 0.41-0.60 moderate; 0.61-0.80 substantial; 0.81-1.00 almost perfect

**Preventing hallucinated categories:**
1. Force LLM to choose from pre-defined code list
2. Require quoted text passages supporting each code
3. Verify quoted passages exist in source (LLMCode does this automatically)
4. Multiple passes, keep only consistent results
5. Temperature = 0

### 5. Cost estimation

| Model | Per posting (10 codes) | 100K postings | With Batch API |
|-------|----------------------|---------------|----------------|
| GPT-4o-mini | ~$0.0012 | ~$120 | ~$60 |
| GPT-4o | ~$0.020 | ~$2,000 | ~$1,000 |
| Claude Haiku | ~$0.0008 | ~$80 | N/A |

### 6. Job posting applications

**Two-stage pipeline:**
1. **Stage 1 (GPT-4o-mini/Haiku):** Extract structured fields, standardize skills, remove boilerplate.
2. **Stage 2 (GPT-4o/Sonnet):** Thematic analysis on cleaned data -- discovering emergent patterns.

### 7. Limitations and pitfalls

**Systematic bias (Ashwin et al. 2025):** LLM coding errors are NOT random -- they correlate with text characteristics. LLMs over-predict sparse codes. Can produce completely incorrect conclusions. Always validate on a gold standard.

**Reproducibility:** Model drift across API versions, temperature > 0 produces different results, prompt sensitivity. Use specific model snapshots, temperature=0, version-control all prompts.

**Practical pitfalls:**
- Context window limits: chunking loses context
- Positional bias: LLMs attend more to beginning/end of prompts
- Order effects: code order in codebook prompts affects results
- Few-shot anchoring: examples anchor too strongly on particular patterns

### 8. Comparison with topic modeling

| Approach | Best For | Limitations |
|----------|----------|-------------|
| **LDA/Topic Modeling** | Quick themes, no API costs, reproducible | Bag-of-words, needs preprocessing, opaque clusters |
| **BERTopic** | Semantic discovery, short text, temporal | Still unsupervised, can produce incoherent topics |
| **LLM Thematic Analysis** | Rich themes, handles nuance, follows codebooks | Expensive, non-deterministic, potential bias |
| **Keyword/Regex** | Known categories, fast, cheap, reproducible | Misses synonyms, paraphrasing, implicit mentions |
| **Fine-tuned Classifier** | High-volume with labeled data, fast inference | Requires training set, doesn't discover new categories |

### 9. Key sources

- Hamalainen et al. (2024). "Scalable Qualitative Coding with LLMs" -- arXiv:2401.15170
- De Paoli (2024). "Performing Inductive TA with an LLM" -- SAGE Journals
- Ashwin, Chhabra, Rao (2025). "Using LLMs for Qualitative Analysis Can Introduce Serious Bias" -- arXiv:2309.17147
- LLMCode Toolkit: https://github.com/PerttuHamalainen/LLMCode
- ESCOX: Skill and Occupation Extraction Using LLMs -- ScienceDirect
- Nesta Skills Extraction Pipeline: https://explosion.ai/blog/nesta-skills

---

# Part B: Statistical and Econometric Methods

## B1. Structural Break Detection and Causal Inference

### 1. Bai-Perron Test

**What it does.** Finds multiple structural breaks at unknown dates in time series. The data tells you when things changed. Uses dynamic programming for efficient global optimization.

**Testing workflow:**
1. **UDmax/WDmax**: Tests null of no breaks against unknown number of breaks. Start here.
2. **Sequential sup F(l+1|l)**: Tests l breaks against l+1. Determines how many breaks.
3. Report confidence intervals for break dates.

**Key assumptions:**
- Linear model within each regime
- Minimum segment length (typically 5-15% of observations)
- HAC standard errors recommended for autocorrelated data
- No trending regressors across breaks

**When assumptions are violated:**
- Short segments: imprecise break dates, low power. Relevant for monthly data where a regime lasts only 6-12 months.
- Gradual transitions: May detect a break in the middle or miss it entirely.

**Labor market applications:** Detecting regime changes in unemployment rates, Beveridge curve shifts, and job posting volume breaks corresponding to major recessions and technology shifts.

---

### 2. Chow Test

**What it does.** Tests for a structural break at a known, pre-specified date. Use as a confirmatory test after Bai-Perron, or for placebo tests.

**How it differs from Bai-Perron:**

| Feature | Chow Test | Bai-Perron |
|---------|-----------|------------|
| Break date | Must be specified | Estimated from data |
| Number of breaks | One | Multiple |
| Data-mining risk | Low (if date is genuine) | Controlled via critical values |
| Simplicity | Very simple | Complex |

**When to use:** When you have a strong a priori reason for the break date (ChatGPT launch, a specific policy).

---

### 3. CUSUM Tests

**What they do.** Detect parameter instability by tracking cumulative sums of recursive residuals. Don't estimate break dates -- just test whether the process is stable.

| Feature | CUSUM | CUSUM-of-Squares |
|---------|-------|-----------------|
| Sensitive to | Mean shifts, gradual drift | Variance changes |
| Best for | Detecting when trend/level changed | Detecting volatility regime changes |

**Use as screening tool:** Run across dozens of occupation categories to flag which ones show instability, then apply Bai-Perron for precise break dating.

---

### 4. Bayesian Online Change Point Detection (BOCPD)

**What it does.** Detects change points in real time as data arrives. Maintains a probability distribution over "run length" (time since last change point). The hazard function encodes prior beliefs about change point frequency.

**Strengths:** Real-time, uncertainty quantification, modular, auto-detects number of change points.
**Weaknesses:** Assumes i.i.d. within regime, O(T^2) without pruning, sensitive to hazard function.

**Best for:** Monitoring dashboards that flag emerging labor market changes in real time.

---

### 5. The `ruptures` Library

The most comprehensive Python library for offline change point detection. Modular: any search algorithm + any cost function.

**Search algorithms:**

| Algorithm | Class | Complexity | Notes |
|-----------|-------|------------|-------|
| Dynamic Programming | `Dynp` | O(Kn^2) | Exact; must specify K breaks |
| PELT | `Pelt` | O(n) to O(n^2) | Exact with penalty; auto-selects K |
| Binary Segmentation | `Binseg` | O(n log n) | Approximate; greedy |
| Bottom-Up | `BottomUp` | O(n log n) | Approximate; agglomerative |

**PELT is usually the best default.**

**Cost functions:**

| Cost | `model=` | Detects changes in... |
|------|----------|-----------------------|
| L2 | `"l2"` | Mean |
| L1 | `"l1"` | Median (robust) |
| RBF kernel | `"rbf"` | Distribution (non-parametric) |
| Linear regression | `"linear"` | Linear relationship |
| Normal MLE | `"normal"` | Mean and variance jointly |

**Practical tips:**
1. Standardize data before kernel-based methods.
2. Set `min_size` to enforce minimum segment length (6 months for monthly data).
3. Use `"rbf"` as default for exploratory analysis.
4. Compare PELT and BinSeg -- agreement increases confidence.

**Choosing the penalty (PELT):**
- BIC: k * log(n) * sigma^2
- AIC: 2k * sigma^2
- Elbow method: plot breaks vs penalty

---

### 6. Interrupted Time Series (ITS)

**What it does.** Estimates the causal effect of an intervention at a known time point. Not just detection -- it's a quasi-experimental design.

**The model:**

```
Y_t = B0 + B1*T + B2*X_t + B3*TX_t + e_t
```

| Coefficient | Interpretation |
|-------------|---------------|
| B0 | Baseline level at T=0 |
| B1 | Pre-intervention slope (existing trend) |
| B2 | **Immediate level change** (did it jump?) |
| B3 | **Slope change** (did the trend bend?) |

Example: If studying whether agent deployment caused a drop in junior posting share:
- B2 < 0: postings dropped immediately
- B3 < 0: decline accelerated over time
- B2 ~ 0, B3 < 0: no immediate effect but gradual erosion

**Power considerations:**
- Minimum ~8 observations pre- and post-intervention
- Slope change (B3) requires more data than level change (B2)
- Positive autocorrelation reduces effective sample size
- ~40 post-intervention months (ChatGPT to early 2026): adequate for both level and slope

**Handling autocorrelation:**
1. Newey-West (HAC) standard errors
2. Prais-Winsten / Cochrane-Orcutt for AR(1)
3. ARIMA errors
4. Seasonal adjustment first

**Pitfall: two parametrizations exist.** Wagner's (more intuitive) has B2 = immediate level change at the intervention point. Bernal's has B2 = difference in intercepts extrapolated to T=0. The immediate effect must be calculated as B2 + B3 * T_intervention. Always verify which you're using.

---

### 7. Markov-Switching Models

**What they do.** Allow parameters to switch between discrete "regimes" according to an unobserved Markov chain. Unlike Bai-Perron (permanent breaks), regimes can recur.

| Feature | Structural Break (Bai-Perron) | Markov Switching |
|---------|-------------------------------|-----------------|
| Transitions | Permanent | Reversible, stochastic |
| Regimes | Estimated endogenously | Must specify number |
| Use case | Permanent structural change | Recurring states (expansion/recession) |

**For job postings:** Two-regime ("normal hiring" vs "contraction") or three-regime ("boom" / "normal" / "recession"). Reveals whether the post-ChatGPT period is a genuinely new regime or normal cyclical behavior.

---

### 8. Difference-in-Differences (DiD)

**What it does.** Compares treatment group (SWE) vs control group before and after treatment. The interaction term estimates the causal effect.

**The parallel trends assumption:** Absent treatment, treated and control groups would have followed the same trajectory. Fundamentally untestable -- it's about a counterfactual.

**Testing parallel trends:**
- Plot pre-treatment trends for both groups
- Run event study with leads/lags; pre-treatment leads should be insignificant
- Caveat (Roth 2022): failing to reject doesn't confirm parallel trends

**The staggered DiD problem:** When units receive treatment at different times, TWFE can produce biased estimates because already-treated units serve as controls for newly-treated units.

**Modern solutions:**
- Callaway & Sant'Anna (2021): Group-time ATT estimates. Python: `differences` package.
- Sun & Abraham (2021): Interaction-weighted estimator. Python: `paneleventstudy`.
- Borusyak, Jaravel, Spiess (2024): Imputation-based approach.

**Labor market application:** Liu, Wang, Yu (2025) used DiD on 285 million Lightcast postings, comparing high vs low AI-substitution occupations after ChatGPT, finding 12% relative decline in high-substitution occupations by 2025.

---

### 9. Synthetic Control Method

**What it does.** Constructs a data-driven counterfactual by finding a weighted combination of control units that best matches the treated unit's pre-intervention trajectory. More defensible than hand-picked controls.

**Inference via placebo tests:**
1. **In-space placebos**: Estimate effect for every control unit. If treated effect is extreme, p-value = 1/(J+1).
2. **Pre/post RMSPE ratio**: Rank across all units. High rank = unusual divergence.
3. **In-time placebos**: Fake intervention in pre-treatment period. Finding an "effect" means method is unreliable.

---

### 10. Event Study Designs

**What they do.** Estimate dynamic treatment effects -- how the effect evolves over time relative to the event. Pre-treatment coefficients serve as a parallel-trends test.

**What to look for:**
- Pre-event coefficients near zero = parallel trends holds
- Immediate but temporary: large at event, decays
- Gradual build-up: grows over time
- Permanent level shift: constant post-event

---

### 11. Decision framework

```
Do you know when the intervention occurred?
├── YES -> Is there a plausible control group?
│   ├── YES -> How many treated units?
│   │   ├── One -> Synthetic Control
│   │   ├── Few -> DiD (possibly with synthetic control weighting)
│   │   └── Many, staggered timing -> Staggered DiD (Callaway-Sant'Anna)
│   └── NO -> Interrupted Time Series
│       ├── Long post-period -> ITS with slope change
│       └── Short post-period -> ITS (level change only)
├── NO -> Do regimes recur or are they permanent?
│   ├── Permanent -> Bai-Perron (offline, retrospective)
│   ├── Recurring -> Markov-Switching
│   └── Unknown -> ruptures (exploratory) then refine
└── Real-time monitoring? -> BOCPD
```

### Comparison table

| Method | Knows break date? | Multiple breaks? | Causal? | Real-time? | Python maturity |
|--------|-------------------|-------------------|---------|-----------|----------------|
| Chow test | Requires it | No | No | No | Good |
| Bai-Perron | Finds it | Yes | No | No | Poor (no native pkg) |
| CUSUM | N/A (screening) | Detects instability | No | Possible | Good (statsmodels) |
| BOCPD | Finds it | Yes | No | **Yes** | Moderate |
| ruptures | Finds it | Yes | No | No | **Excellent** |
| ITS | Requires it | No | **Quasi-causal** | No | Good (statsmodels) |
| Markov switching | N/A | Yes (recurring) | No | Possible | **Excellent** (statsmodels) |
| DiD | Requires it | One or staggered | **Yes (if PT holds)** | No | Good |
| Synthetic control | Requires it | One | **Yes (if fit good)** | No | Good |
| Event study | Requires it | Dynamic effects | **Yes (if PT holds)** | No | Moderate |

### For job posting analysis specifically

**"Did ChatGPT cause a structural break?"**
1. ruptures (PELT, RBF kernel) on overall time series
2. Bai-Perron (via R/rpy2) for formal testing
3. ITS to estimate magnitude of level and slope change

**"What was the causal effect?"**
1. DiD: high-AI-exposure vs low-AI-exposure occupations
2. Event study for parallel trends check
3. Staggered DiD (Callaway-Sant'Anna) if differential timing
4. Synthetic control for single-occupation deep dives

### Key references

**Structural breaks:**
- Bai & Perron (1998, 2003). Multiple structural changes. *Econometrica* / *J Applied Econometrics*
- Truong, Oudre, Vayatis (2020). Offline change point detection. *Signal Processing*
- Adams & MacKay (2007). Bayesian online changepoint detection. *arXiv:0710.3742*

**Causal inference:**
- Callaway & Sant'Anna (2021). DiD with multiple time periods. *J Econometrics*
- Goodman-Bacon (2021). Variation in treatment timing. *J Econometrics*
- Roth (2022). Pretest with caution. *AER: Insights*
- Liu, Wang, Yu (2025). Labor demand in the age of generative AI. SSRN.

### Python library reference

| Library | Install | Purpose |
|---------|---------|---------|
| `ruptures` | `pip install ruptures` | Change point detection |
| `statsmodels` | `pip install statsmodels` | ITS, CUSUM, Markov switching, OLS, GLM |
| `differences` | `pip install differences` | Callaway-Sant'Anna staggered DiD |
| `pysyncon` | `pip install pysyncon` | Synthetic control |
| `SyntheticControlMethods` | `pip install SyntheticControlMethods` | Alternative synthetic control |
| `paneleventstudy` | `pip install paneleventstudy` | Sun-Abraham event studies |
| `bayesian_changepoint_detection` | GitHub: hildensia | BOCPD |
| `chowtest` | `pip install chowtest` | Chow test |
| `linearmodels` | `pip install linearmodels` | Panel data with FE |

### Online resources

- [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/)
- [Causal Inference: The Mixtape](https://mixtape.scunning.com/)
- [The Effect](https://theeffectbook.net/)
- [ruptures documentation](https://centre-borelli.github.io/ruptures-docs/)
- [Gregory Gundersen's BOCPD tutorial](https://gregorygundersen.com/blog/2019/08/13/bocd/)

---

## B2. Text Embeddings and Classification

### 1. Sentence Transformers (SBERT)

**What it does.** Maps text to fixed-size dense vectors where semantic similarity = vector proximity. Foundation for clustering, drift measurement, and similarity analysis. Finding the most similar pair in 10K sentences: raw BERT ~65 hours, SBERT ~5 seconds.

**Key models:**

| Model | Dimensions | Speed | Quality | When to use |
|-------|-----------|-------|---------|-------------|
| `all-MiniLM-L6-v2` | 384 | 5x faster | Good (~84-85% STS-B) | Prototyping, large-scale search |
| `all-mpnet-base-v2` | 768 | Baseline | Best general (~87-88% STS-B) | Production quality |
| `intfloat/e5-large-v2` | 1024 | Slower | State-of-the-art MTEB | Maximum quality with GPU |

**Common pitfalls:**
- Max sequence length is typically 256-512 tokens. Job postings often exceed this. Chunk and average, or use the most informative section.
- Generic models miss domain-specific semantics. Domain adaptation (JobBERT) addresses this.
- Quality degrades on very short (just a title) and very long (full posting) inputs. Sweet spot is 1-3 sentences.

---

### 2. Domain-adapted embeddings: JobBERT, JobSpanBERT, SkillBERT

**Why generic embeddings underperform on job text.** Pre-trained models learned from Wikipedia/BookCorpus, not "5+ YOE in distributed systems." The distributional semantics of job postings are different.

**Key models:**

**JobBERT v2** (Decorte et al., 2025):
- Built on `all-mpnet-base-v2`, trained on 5.5M+ job title-skills pairs
- Max sequence length: 64 tokens (optimized for titles, not descriptions)
- 1024-dimensional embeddings
- Achieves 0.6457 MAP on TalentCLEF benchmark

**JobSpanBERT** (Zhang et al., 2022): SpanBERT + domain adaptation. Best for extracting multi-token skill spans. F1 56.64 on SkillSpan.

**SkillBERT**: BERT trained from scratch on skills from recruitment records. Classifies skills into competency groups.

**When to use which:**
- Job titles (matching, normalization, clustering): JobBERT v2
- Skill extraction as spans: JobBERT or JobSpanBERT as NER backbone
- Full job descriptions: Generic sentence transformers may be fine (longer text compensates)
- Custom pipeline: Do your own continued pre-training on your job posting corpus

**Failure modes:**
- Domain-adapted models can overfit to training distribution (US tech vs European manufacturing)
- JobBERT v2's 64-token limit: designed for titles, not descriptions
- Noisy training data: postings list aspirational skills, duplicates, omissions

---

### 3. Cosine similarity

**What it measures.** The cosine of the angle between two vectors. Measures direction, not magnitude. Scale-invariant.

**Interpreting scores for job postings:**

| Score Range | Interpretation | Example |
|------------|----------------|---------|
| 0.85-1.0 | Near-duplicates | Same job reposted by two agencies |
| 0.70-0.85 | Highly similar | "Senior Python Developer" vs "Lead Python Engineer" |
| 0.50-0.70 | Related but distinct | "Data Scientist" vs "ML Engineer" |
| 0.30-0.50 | Loosely related | "Software Engineer" vs "Product Manager" |
| < 0.30 | Unrelated | "Nurse" vs "Software Engineer" |

Thresholds shift by model. Domain-adapted models produce higher within-domain similarities and sharper between-domain separations.

**Failure modes:**
- Treats all dimensions equally (learned metrics can outperform for specific tasks)
- Short, generic texts score high without meaningful similarity
- "Hubness" problem in high dimensions

---

### 4. Embedding drift / trajectory analysis

**Core idea.** Embed job postings from different time periods into the same vector space. Track how the meaning of job categories shifts over time.

#### Centroid tracking

**Limitation:** Centroids average out internal variation. A category might split without the centroid moving.

#### Distribution comparison: MMD

Maximum Mean Discrepancy tests whether two sets of embeddings come from the same distribution.

#### Domain classifier drift detection

Train a binary classifier to distinguish "reference period" from "current period" embeddings. ROC AUC >> 0.5 means distributions are different. The AUC value measures drift magnitude. Most robust approach across different embedding models.

---

### 5. Dimensionality reduction for visualization

| Feature | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| Speed | Fast | Slow (>10K pts) | Fast |
| Global structure | Preserved | Lost | Mostly preserved |
| Local structure | Approximate | Excellent | Excellent |
| Deterministic | Yes | No | No (seedable) |
| New point projection | Yes | No | Yes |
| Axes interpretable | Yes | No | No |
| Best for | Quick overview, preprocessing | Small dataset visualization | Default choice |

**t-SNE critical parameter: perplexity.** 30-50 for job postings. Axes are arbitrary and uninterpretable. Only cluster existence and within-cluster structure are meaningful.

---

### 6. Document similarity matrices

**Applications:**
1. Duplicate detection (threshold at 0.90+)
2. Hierarchical clustering for natural job families
3. Cross-temporal comparison (2020 rows vs 2026 columns for same title)
4. Market segmentation via spectral clustering on similarity graph

**Scalability:** NxN matrix for 1M postings = ~4TB. Use approximate nearest neighbors (FAISS, Annoy, ScaNN) for top-k per posting instead.

---

### 7. Fine-tuned BERT classifiers

**When you need high accuracy with 1000+ labeled examples per class.**

**Critical hyperparameters:**
- Learning rate: 2e-5 to 5e-5 (higher = catastrophic forgetting)
- Epochs: 3-5
- Batch size: 16-32
- Warmup: 5-10% of training steps

**Failure modes:**
- Class imbalance (far more mid-level than C-suite). Use class weights or focal loss.
- 512-token limit truncates long postings. First 512 usually most discriminative.
- Distribution shift: 2020-trained classifier may misclassify 2026 postings.

---

### 8. SetFit (few-shot classification)

**What it does.** Achieves competitive classification with 8-16 labeled examples per class. No prompts, no LLMs needed.

**Two phases:**
1. Contrastive fine-tuning: generate pairs from labeled set, fine-tune sentence transformer. 32 examples yield ~496 training pairs.
2. Classification head: logistic regression on fine-tuned embeddings.

**Performance:** SetFit (355M params) outperforms GPT-3 (175B) on RAFT benchmark. 28x faster to train than T-Few.

**Failure modes:**
- Example selection matters enormously. Choose diverse, representative examples.
- Struggles with semantically close classes ("mid-level" vs "senior").
- Degrades with 20+ classes. Consider hierarchical approaches.

---

### 9. Zero-shot classification

**What it does.** Classifies text without training data by repurposing NLI models. The input becomes the "premise"; each label becomes a hypothesis ("This is a job posting for a {label} position").

**Key models:**
- `facebook/bart-large-mnli`: Fast, good general performance
- `microsoft/deberta-v2-xlarge-mnli`: Highest accuracy, slow
- `cross-encoder/nli-deberta-v3-base`: Good quality/speed trade-off

**Hypothesis template matters.** Domain-specific templates dramatically improve accuracy.

**Strengths:** No training data, labels changeable on the fly, 70-80% accuracy for broad categories.
**Weaknesses:** Lower ceiling than fine-tuned, sensitive to template wording, struggles with domain-specific labels.

---

### 10. Seniority classification from job text

Seniority is encoded implicitly: years of experience, scope of responsibility, skill complexity, reporting structure.

**Approach:**
1. Train classifier on 2020-2022 postings (title-based heuristics + manual review)
2. Apply to 2026 postings
3. When a 2026 "Junior" posting classifies as "senior" by 2020 standards, that's evidence of credential creep

**Failure modes:** Job titles are unreliable labels (startup "Senior" vs Google "Senior"). Industry effects. Model may learn surface features (length, formatting) vs substance.

---

### 11. Multi-label classification

For postings spanning multiple skill categories. Uses sigmoid per class (independent) instead of softmax (competitive).

**Threshold selection:** Fixed 0.5, per-class tuned on validation, or micro-averaged F1 optimization.

---

### 12. Active learning

Maximizes classifier accuracy per labeled example by choosing which examples to label next.

**Uncertainty sampling strategies:**
- **Least confidence:** Select where max predicted class probability is lowest
- **Margin:** Select where gap between top two classes is smallest
- **Entropy:** Select highest prediction entropy. Usually best for multi-class.

**Performance:** Active learning with BERT achieves F1 0.91 vs 0.85 under random sampling with same labeling budget.

**Pitfalls:** Sampling bias (overselects unusual examples; periodically add random samples), cold start (use random for first 50-100, then switch).

---

### 13. Skill extraction (NER-based)

**SkillSpan benchmark:** 391 postings, 14,538 sentences, 232,220 tokens, inter-annotator agreement 0.70-0.75.

**Performance (F1 on SkillSpan):**

| Model | Skills F1 | Knowledge F1 |
|-------|-----------|-------------|
| BERT-base | ~48 | ~57 |
| JobBERT (DAPT) | ~54 | **~64** |
| JobSpanBERT | **~57** | ~62 |

**Pitfalls:** Sensitive to formatting (normalize text), span boundary errors, English-only.

---

### 14. Taxonomy mapping (ESCO/O*NET)

**Challenges:** Taxonomy descriptions don't match employer language, granularity mismatch, new skills emerge faster than updates, many-to-many mappings.

**Practical recommendation:** Hybrid approach. Extract with NER/LLM. Map to ESCO/O*NET what you can. Cluster unmapped skills to discover emerging categories.

---

### 15. Decision framework

| Research Question | Primary Technique | Supporting |
|------------------|-------------------|-----------|
| "How has junior SWE changed?" | Embedding drift + Seniority classifier | UMAP, similarity matrices |
| "What skills are emerging?" | Taxonomy-free extraction + Clustering | Zero-shot for validation |
| "Classify 100K postings by seniority" | SetFit or fine-tuned BERT | Active learning |
| "Which skills does this posting require?" | NER extraction + taxonomy mapping | Multi-label classification |
| "Are two job categories converging?" | MMD or Wasserstein on embeddings | Centroid tracking, UMAP |
| "Quick label 500 with no training data" | Zero-shot classification | Then upgrade to SetFit |

### Model selection cheat sheet

| Budget | Classification | Embeddings | Skill Extraction |
|--------|---------------|------------|-----------------|
| No labeled data | Zero-shot (bart-large-mnli) | all-mpnet-base-v2 | LLM prompting |
| 8-16 labels/class | SetFit | all-mpnet-base-v2 | LLM prompting |
| 100+ labels/class | Fine-tuned BERT | Domain-adapted (JobBERT v2) | Fine-tuned NER (JobBERT) |
| 1000+ labels/class | Fine-tuned DeBERTa | Domain-adapted + custom DAPT | Fine-tuned NER + CRF |

### Key sources

- Reimers & Gurevych (2019). Sentence-BERT. *EMNLP*. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
- Decorte et al. (2021). JobBERT. FEAST @ ECML-PKDD.
- Decorte et al. (2025). Efficient text encoders for labor market analysis. *IEEE Access*.
- Zhang et al. (2022). SkillSpan. *NAACL*. [arXiv:2204.12811](https://arxiv.org/abs/2204.12811)
- Tunstall et al. (2022). SetFit. [arXiv:2209.11055](https://arxiv.org/abs/2209.11055)
- Bhola et al. (2020). Retrieving Skills from Job Descriptions: XMLC. *COLING*.
- Sentence Transformers: https://sbert.net/
- TechWolf/JobBERT-v2: https://huggingface.co/TechWolf/JobBERT-v2
- ESCO: https://esco.ec.europa.eu/en/classification
- O*NET: https://www.onetcenter.org/taxonomy.html

---

## B3. Regression and Econometric Methods

### 1. OLS with text-derived features

**What it does.** Regress outcomes (wage, application count) on text-extracted features (skill counts, keyword frequencies, topic shares, embedding features).

**The measurement problem (Ash & Hansen 2023):** Text quantification and econometric models are usually treated separately. The downstream regression ignores uncertainty in upstream text measurement. Valid inference requires bias correction or joint estimation.

**Standard errors -- what to use:**

| Type | When | Python |
|------|------|--------|
| HC1 (White's with df correction) | Default robust SE. Most common in applied economics | `model.fit(cov_type='HC1')` |
| HC3 | Best for smaller samples. Recommended default | `model.fit(cov_type='HC3')` |
| Clustered by firm | Postings from same firm share unobservables | `model.fit(cov_type='cluster', cov_kwds={'groups': df['firm_id']})` |
| Clustered by metro | Local labor market shocks affect all postings | `model.fit(cov_type='cluster', cov_kwds={'groups': df['msa']})` |
| Two-way clustering | Both firm and time dimensions matter | Use `linearmodels` |

**Pitfalls:**
- Text-derived features without measurement error acknowledgment cause attenuation bias
- Failing to cluster when observations share shocks = spuriously small p-values
- Too many correlated text features = multicollinearity

---

### 2. Panel data methods

**Why fixed effects matter.** Firms have persistent unobserved characteristics (culture, compensation philosophy) that correlate with both skill requirements and wages. Fixed effects remove all time-invariant unobserved heterogeneity.

**Types of fixed effects for job posting data:**
- Firm FE: persistent firm characteristics
- Time FE: aggregate trends (economy-wide AI adoption)
- Metro area FE: local labor market conditions
- Occupation FE: inherent occupational differences
- Firm x time interactive FE: firm-specific time trends

**Pitfalls:**
- Cannot estimate time-invariant variables with entity FE
- TWFE can mislead with heterogeneous treatment effects and staggered adoption
- Short panels + nonlinear models = incidental parameters problem

---

### 3. Quantile regression

**What it does.** Estimates conditional quantiles instead of the conditional mean. Reveals how effects vary across the distribution.

**When to use:** When effects differ across the distribution. AI skill requirements might affect low-wage postings differently than high-wage.

---

### 4. Count models

**For non-negative integer outcomes:** number of skills per posting, AI keyword counts, postings per firm-quarter.

**Model selection:**
1. Start with Poisson. Check for overdispersion (variance >> mean).
2. If overdispersed, use negative binomial.
3. If excess zeros beyond what count model predicts, consider zero-inflated variants.

**Pitfall:** Poisson is consistent for the conditional mean even under misspecification, but use robust SEs. Don't use OLS for count data.

---

### 5. Logistic / probit regression

**For binary outcomes:** Does this posting mention AI? Is this a junior role?

Report average marginal effects (AME), not raw coefficients. In nonlinear models, marginal effects depend on all other variables.

**LPM (OLS on binary outcome)** is widely used in applied economics because coefficients are directly interpretable as marginal effects and easily accommodate fixed effects. Use robust SEs.

---

### 6. Index construction

#### Skill Breadth Index

Higher entropy = more evenly distributed across categories = broader skill profile.

#### Senior Keyword Infiltration Index

```
Infiltration_t = (1/N_junior_t) * SUM (count of senior keywords in posting i) / (total keywords in posting i)
```

Define senior keywords using chi-squared or log-odds ratio against baseline period.

#### Archetype Shift Index

```
ASI_t = mean(AI-orchestration keyword count) / mean(Management + AI-orchestration keyword count)
```

Values range 0 (purely management) to 1 (purely AI-orchestration).

**Robustness:** Document keyword dictionaries, run sensitivity analyses with alternatives, validate against manual coding.

#### Composite indices

Equal weighting is a good baseline. PCA maximizes variance but variance is not the same as capturing your theoretical construct.

---

### 7. Survival / duration analysis

**Time-to-fill as a labor market signal.** Vacancy duration is countercyclical. ~15% of vacancies take > 3 months. Firms vary skill requirements over the business cycle.

**Pitfalls:** Ignoring censoring biases toward shorter durations. Always test proportional hazards assumption with Schoenfeld residuals.

---

### 8. Descriptive statistics

#### Gini coefficient / HHI

HHI = SUM(s_i^2) where s_i is market share. Ranges from 1/N to 1.

#### Jensen-Shannon Divergence

Symmetric, bounded [0,1], always defined. Track JSD(junior_t, senior_t) over time -- decreasing = convergence.

#### Kolmogorov-Smirnov test

Detects any distributional difference (not just means). With large datasets, even trivial differences are "significant" -- always report effect sizes.

#### Effect sizes

- **Cohen's d**: 0.2 small, 0.5 medium, 0.8 large. Assumes normality.
- **Cliff's delta**: Non-parametric, -1 to 1. Better for skewed data (skill counts, wages).

---

### 9. Network / graph methods

#### Skill co-occurrence networks

Key reference: PLOS Complex Systems (2024), 65M UK postings, 3,906 skills, 21 robust clusters.

**Key findings:** No skill silos. Specialist skills premium (negative correlation between closeness centrality and salary, rho = -0.51). Skills per advert increased from 8.60 to 10.73 over 2016-2022.

**Centrality metrics:**
- **Degree**: How many skills does this co-occur with? (generalist skills)
- **Betweenness**: Does this bridge clusters? (connector skills)
- **Closeness**: How close to all others? Low closeness = specialist = higher wages.

**Community detection:** Louvain (fast, NetworkX built-in), Leiden (improved, `leidenalg` package), Markov Stability (sophisticated, `PyGenStability`).

#### Temporal network analysis

Build separate networks per time period. Compare community structure, centrality, edge density. Track individual skills' centrality trajectories.

**Metrics:** NMI between community assignments (low = major reorganization), edge volatility, centrality trajectories.

---

### 10. Multiple testing and robustness

#### Multiple testing corrections

- **FWER (Holm)**: When any false positive is costly.
- **FDR (BH)**: When screening many hypotheses. Standard for large-scale text analysis.

#### Placebo tests

1. **Temporal**: Run analysis at multiple pre-treatment dates. "Effects" at placebo dates = your method detects noise.
2. **Outcome**: Apply to outcomes that shouldn't be affected. AI skill changes for plumbers?
3. **Keyword**: Test whether random keywords show same temporal patterns.

Dreber et al. (2024) found strong evidence of selective underreporting of significant placebo tests. Report all placebos.

#### Sensitivity analysis

Vary: keyword dictionaries, classification thresholds, preprocessing, time periods, geographic granularity. Present results across specifications -- if sign and magnitude are stable, the result is robust.

#### Bootstrap confidence intervals

Use for any statistic without analytical standard errors: custom indices, JSD, Gini, network metrics.

---

### 11. Python library reference

| Library | Purpose | Key Classes/Functions |
|---------|---------|----------------------|
| `statsmodels` | OLS, GLM, logit/probit, quantile regression, count models | `OLS`, `GLM`, `Logit`, `QuantReg`, `NegativeBinomial` |
| `linearmodels` | Panel data (FE, RE) | `PanelOLS`, `RandomEffects` |
| `lifelines` | Survival analysis | `KaplanMeierFitter`, `CoxPHFitter`, `logrank_test` |
| `networkx` | Graphs, centrality, communities | `Graph`, `louvain_communities`, `bipartite` |
| `leidenalg` | Leiden community detection | `find_partition`, `CPMVertexPartition` |
| `scipy.stats` | KS test, bootstrap, entropy | `ks_2samp`, `bootstrap`, `entropy` |
| `scipy.spatial.distance` | Jensen-Shannon divergence | `jensenshannon` |
| `sklearn.decomposition` | PCA, factor analysis | `PCA`, `FactorAnalysis` |
| `statsmodels.stats.multitest` | Multiple testing corrections | `multipletests` |

---

### 12. Decision guide

| Research Question | Primary Method | Complement With |
|---|---|---|
| Do AI skills predict higher wages? | OLS with robust SEs | Panel FE, quantile regression |
| How has skill demand changed? | Panel FE (firm + time) | JSD, KS test |
| Different effects for high vs. low-wage? | Quantile regression | Cohen's d / Cliff's delta |
| How many AI skills per posting? | Negative binomial / ZINB | Gini for concentration |
| Is this posting AI-relevant? | Logit / probit | LPM with FE as robustness |
| Junior and senior converging? | JSD over time | KS test, bootstrap CIs |
| What skill clusters exist? | Co-occurrence network + Louvain | Bipartite projection, PCA |
| How long do AI postings take to fill? | Cox PH model | Kaplan-Meier by group |
| Is the result robust? | Sensitivity analysis | Placebo tests, BH correction, bootstrap |

### Key references

- Deming and Kahn (2018). Skill requirements across firms. *J Labor Economics*. [45M ads, wage regressions]
- Hershbein and Kahn (2018). Recessions and upskilling. *AER*. [Great Recession, vacancy postings]
- Ash and Hansen (2023). Text algorithms in economics. *Annual Review of Economics*.
- Hampole et al. (2025). AI and the labor market. [GTE embeddings, Lightcast skills]
- PLOS Complex Systems (2024). Co-occurrent skills in UK job adverts. [65M postings, 21 clusters]
- Dreber et al. (2024). Selective reporting of placebo tests. *Economic Inquiry*.
