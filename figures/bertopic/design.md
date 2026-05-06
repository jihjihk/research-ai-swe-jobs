# BERTopic analysis — design doc

**Status:** draft, pre-registration target. Owner: Jihyun + Gabor. Conference target: AIES 2026.

This document specifies the BERTopic analysis that underpins the paper's "role landscape" claims (role families emerging, legacy stacks shrinking, within-family vocabulary drift, role concentration). BERTopic is used twice: once on an **exploration sample** to inform the L1/L2 taxonomy, and once on a disjoint **confirmation sample** to discover and validate what's in the data. The doc is meant to be read end-to-end before any clustering script is written, and the exploration vs confirmation split must be respected — once L1/L2 are frozen on the exploration sample, no information from the confirmation sample flows back to retune them.

The doc is split into thirteen sections: what BERTopic is doing in this paper (§1), inputs (§2), sample (§3), method (§4), cluster naming (§5), validation (§6), ablations (§7), deliverables (§8), risks (§9), pre-registration (§10), runtime (§11), implementation plan (§12), reproducibility and code release (§13). The most decision-critical section to read first is §1.4 — what claims this analysis can prove or disprove.

---

## 1. Purpose and scope

### 1.1 Where BERTopic sits in the methodology

The paper has three classification layers, and BERTopic appears at two distinct moments in the analytical timeline.

| Layer | Purpose | Method | Vocabulary |
|---|---|---|---|
| **L1 — Role family** | Assign each posting to one of ~12 role families (Frontend, Backend, AI/LLM Engineer, …) | One LLM call per posting against fixed prompt | **Closed**, frozen pre-registered |
| **L2 — Skill tags** | Tag binary skill matrix (~50 skills: RAG, agent orchestration, K8s, mentorship, …) | One LLM call per posting | **Closed**, frozen pre-registered |
| **L3 — BERTopic** | Discover what's in the data without imposing a list | UMAP → HDBSCAN → c-TF-IDF | **Open** |

L1 and L2 answer questions we can pre-register ("did the AI/LLM Engineer family grow?"). L3 answers ones we cannot ("are our families actually distinct density regions in description space?", "what falls outside our list?").

**Exploration / confirmation split — the spine of this analysis.** BERTopic is used in two distinct roles, on two disjoint samples, in two phases:

- **Phase 1 — taxonomy-development BERTopic (exploration set).** Fit BERTopic on the **exploration sample** (asaniczka 2024 only — see §3.1). Use the resulting clusters, KeyBERTInspired top n-grams, and c-TF-IDF terms as one input alongside the literature draft when shaping L1 and L2. Output: a revised, frozen L1/L2 taxonomy committed before any LLM classification runs on the confirmation sample.
- **Phase 2 — discovery-and-validation BERTopic (confirmation set).** Refit BERTopic from scratch on the **confirmation sample** (arshkon 2024 + scraped 2026). Use the result for the paper's discovery claims (§1.2) and as the load-bearing convergent-validity test of L1 — because the confirmation sample's data was *not* used to shape L1, the test is non-circular.

This split is the difference between a credible pre-registration and a fictitious one. If the same data both built and tested the taxonomy, the test is empty. The exploration set buys empirical grounding without circularity. The methodology cost is that the in-sample within-2024 placebo (asaniczka vs arshkon, side-by-side) becomes weaker as a within-period source-confound check, since asaniczka has been read; we mitigate by reporting arshkon-only structure as the held-out 2024 reference and using exploration-vs-confirmation centroid alignment as a stronger version of the placebo (§6.6).

L3 (Phase 2) is also our concentration / entropy instrument: by measuring how confirmation-sample postings distribute over data-driven clusters in 2024 vs 2026, we can claim "roles are concentrating" without that claim depending on the L1 taxonomy being right.

### 1.2 The four claims BERTopic must support

Named here for cross-reference; falsification conditions and predicted signatures are in §1.4.1.

1. **C1 — Discovery.** AI-native role families have crystallized.
2. **C2 — Decline.** Specific legacy stacks (.NET, COBOL, mainframe, PHP/WordPress, AUTOSAR, ServiceNow, PLC) are shrinking.
3. **C3 — Vocabulary drift.** Existing role clusters are being rewritten in place even when their posting share is stable (DevOps → platform / reliability; QA → CI/CD / integration; cloud → ML / agents).
4. **C4 — Concentration.** Effective number of clusters decreases from 2024 to 2026.

Plus four theoretical-framework claims (T1–T4 in §1.4.2) that BERTopic-derived numbers feed without being the sole evidence for.

### 1.3 What BERTopic is *not* doing

- Not the sole source of the L1 role-family labels. The labels are drafted from the literature first (Indeed Hiring Lab, Burning Glass, LaborSpace, SOC); Phase 1 BERTopic on the exploration sample adds, merges, or splits families with documented rationale; once frozen, labels are assigned to *all* postings (including confirmation) by the L1 LLM. Phase 2 BERTopic on the confirmation set then asks "do these families recover from the held-out data, and what falls outside them?"
- Not the sole source of L2 skill tags. Same exploration/confirmation logic: literature draft plus exploration-set BERTopic n-gram and KeyBERTInspired suggestions, frozen, then applied.
- Not the source of skill or family counts in the paper — those are L1/L2 LLM-classified counts.
- Not generating direct claims about juniors vs seniors — those go through L1/L2 conditioned on `seniority_final`. BERTopic supplies a **topics-per-class** view as a triangulation, not a primary instrument.
- Not the source of any AI-rate claim. AI rate uses the canonical `AI_VOCAB_PATTERN` regex on `description_core_llm`. BERTopic only segments where that AI vocabulary lives.

### 1.4 What this analysis can prove or disprove

The point of laying this out before any code runs is to commit to falsification conditions. Each row below names a load-bearing claim from the paper, the BERTopic signature we'd expect if it's true, and the signature that would force us to walk it back. Both columns are paper-visible: we report whichever way each row lands.

#### 1.4.1 Primary claims (paper §"SWE is changing")

| Claim | Predicted BERTopic signature | Falsification signature |
|---|---|---|
| **C1 — AI-native role families have crystallized.** | ≥ 1 cluster whose c-TF-IDF top terms are dominated by AI/LLM/agent vocabulary, n_2024 small (< 5% of period total), n_2026 large (> 15%), and labeled by both reviewers as "AI engineer" or a clear sub-archetype. | No cluster's top vocabulary is AI-flavored; or n_2026/n_2024 < 2× for the AI-flavored cluster(s); or the cluster exists in 2024 at comparable size and has only relabeled itself. |
| **C2 — Specific legacy stacks are shrinking.** | ≥ 1 cluster whose vocabulary is dominated by `.NET`, `COBOL`, `mainframe`, `PHP`/`WordPress`, `AUTOSAR`, `ServiceNow`, or `PLC`, with n_2026/n_2024 < 0.6 and reviewer-named accordingly. | Posting share for these stacks is flat or grows; or the vocabulary doesn't form discrete clusters at all (legacy work has dispersed into other clusters rather than shrinking as a class). |
| **C3 — Existing roles are being rewritten in place.** | ≥ 4 stable clusters (n_2024 ≈ n_2026 within 30%) whose c-TF-IDF top-20 changes by ≥ 30% set-difference between periods (entered minus exited terms). | Stable clusters have stable vocabulary (set-difference < 15%) — change is happening only at the cluster level, not within-cluster level. The "rewriting in place" framing dies; we tell only the cluster-emergence story. |
| **C4 — Roles are concentrating.** | Effective number of clusters (entropy of share distribution at the headline K) lower in 2026 than 2024 by ≥ 10%, **OR** top-5 cluster share (HHI) higher by ≥ 5pp. The two metrics should agree in direction. | Cluster fragmentation increases or stays flat; the "consolidation" framing in the paper outline doesn't survive and we drop it. |

If C1 fires while C2, C3, or C4 do not, the paper has a one-claim narrative ("AI-engineering boom") not a four-claim one. We commit to reporting whichever subset survives, not pretending the whole frame holds.

#### 1.4.2 Theoretical-framework claims (paper §"State of transition")

| Claim | Predicted BERTopic signature | Falsification signature |
|---|---|---|
| **T1 — Brynjolfsson J-curve / intangible-investment phase.** | AI-flavored clusters' 2026 vocabulary is future-tense and exploratory ("will use," "exploring," "building toward," "investigating"); high posting volume but low specificity per posting. | AI vocabulary is concrete and present-tense; AI is integrated alongside other technologies as a routine tool, not as a forward bet. |
| **T2 — Seniority-biased technological change reverses (Hosseini & Lichtinger).** | Junior-leaning clusters shrink less than senior-leaning clusters in routine-coding cells; AI/LLM clusters skew senior in `topics_per_class(seniority_final)`. | Seniority distribution flat across clusters, or junior clusters shrink fastest — the SBTC-reversal framing fails. |
| **T3 — Senior bottleneck persists (Autor & Thompson; Gans & Goldfarb "O-Ring").** | New AI clusters have higher senior share than the corpus average; orchestration / mentorship vocabulary concentrates in senior YoE bins inside those clusters. | New clusters are junior-balanced or junior-tilted — the "AI lifts the codified rungs but seniors persist" thesis loses its support. |
| **T4 — Normal-technology diffusion (Narayanan & Kapoor).** | 2024→2026 cluster restructuring is gradual: many partial movements (vocabulary drift in many clusters, growth/shrinkage at moderate rates) rather than a few discontinuous jumps. | A small number of clusters absorb almost all change; the rest are static — a "discontinuous shock" reading rather than a diffusion reading. |

#### 1.4.3 Pessimist-vs-skeptic resolution (paper §"The losers of transition")

The paper's framing names a public disagreement (Karpathy/Amodei "juniors are cooked" vs Garman/Rachitsky "junior demand fine"). BERTopic-testable angles:

- **Junior cluster mix:** topics-per-class on `seniority_final` shows whether the junior posting **mix** in 2026 differs from 2024 even when junior **share** is flat. If the same proportion of postings is junior but those juniors live in different clusters, that's "scope changes, not headcount" — the paper outline's pre-stated finding. If both share and mix are stable, the displacement narrative weakens further; if share is flat but mix has shifted heavily, both sides of the debate are partly right.
- **AI vocabulary diffusion vs concentration:** if AI vocabulary rises across **all** clusters (including ones with no AI core), that's "AI talk is cheap" — descriptions are being touched up. If AI vocabulary stays concentrated in dedicated clusters, that's "AI work is real" — there's a coherent specialty, not just rebranding.

#### 1.4.4 The AI-cohort sub-structure question (continuation of v3 prior)

Open from `composite_B_v3_findings.md`: is "AI Engineer" splitting into RAG / agents / evals / foundation-model research, or is it still one undifferentiated job? The v3 prior found it had not yet differentiated geometrically (silhouette 0.27, bimodal seed-ARI). This run tests whether four more weeks of 2026 data plus OpenAI 3072-d embeddings change the answer:

- **Differentiated:** hierarchical clustering inside the AI cluster(s) produces ≥ 3 sub-clusters, cross-scale silhouette ≥ 0.4, reviewers name them distinctly with high agreement.
- **Not yet differentiated (v3-consistent):** one mega-cluster, possibly with vertical (healthcare, fintech) splits the geometry enforces but no horizontal (RAG/agents/evals) splits it can find.

A null result here is itself a finding for the paper: it says "the field is naming a specialty before the work has split into sub-specialties."

#### 1.4.5 What this analysis cannot prove

Stated up front, before claims, so reviewers cannot accuse us of conflating registers. BERTopic on job descriptions speaks to:

- The **filter** (what employers write down)
- The **vocabulary** (what they call it)
- The **distribution** (how many of each)

It does **not** speak to:

- Realized hiring (postings ≠ hires; ghost postings exist)
- Wages or salary distributions
- Actual on-the-job content (descriptions ≠ work; cf. METR slowdown vs Peng et al. speedup, both out of scope here)
- Causation between AI capability and posting change (correlation only)
- Counterfactuals ("what would have happened without AI") — the analysis is descriptive
- Cross-platform validity (LinkedIn-only)

Every causal claim in the paper must reroute through the JD-substrate framing the broader methodology section establishes; BERTopic results are a richer description, not a different epistemology.

---

## 2. Inputs

### 2.1 Source

`data/unified_core.parquet` — already preprocessed, LLM-classified, embedded.

| Field | Type | Use |
|---|---|---|
| `uid` | str | Row identifier; cluster labels keyed on this |
| `source` | str | `kaggle_arshkon` / `kaggle_asaniczka` / `scraped` |
| `period` | str | `2024-01`, `2024-04`, `2026-03`, `2026-04` |
| `title` | str | Used in cluster characterization, *not* in embedding (already in `description_core_llm` via Stage 11 input concat) |
| `description_core_llm` | str | LLM-stripped boilerplate-removed body — the c-TF-IDF substrate |
| `job_description_embedding` | float32[3072] | OpenAI `text-embedding-3-large`, computed at Stage 11 from `title + description_core_llm` |
| `is_swe`, `is_control` | bool | Sample selection |
| `seniority_final`, `yoe_min_years_llm` | various | Post-hoc characterization, never input to clustering |
| `company_name_canonical`, `is_aggregator`, `metro_area` | str/bool | Per-firm cap; characterization |
| `date_flag`, `has_llm_extraction`, `has_llm_classification` | various | Filter to clean rows |

### 2.2 Embedding choice — why OpenAI 3072-d, not MiniLM 384-d

The prior BERTopic run (`eda_archive/scripts/S27_v2_bertopic.py`, archived) used `all-MiniLM-L6-v2` 384-d. It produced a usable taxonomy (29 families) but had two known failure modes documented in `composite_B_v3_findings.md`:

- **Mega-cluster blending.** A single density region housed both AI/ML and traditional data-science postings; embeddings could not separate them, requiring a post-hoc AI-vocab regex split (`0_AI` / `0_nonAI`).
- **Stability ARI 0.44–0.49 across seeds** at family granularity — seed-sensitive enough that one of three seeds produced a materially different layout.

`text-embedding-3-large` at 3072-d is the standard upgrade and is already computed. Two reasons it should help: (a) larger capacity for fine semantic distinctions in technical vocabulary; (b) trained on substantially more code/tech text than MiniLM. The hypothesis it should test is whether the AI/data mega-cluster splits without regex assistance. We should not assume it does — we report whichever way it lands.

We **also re-run the entire pipeline with MiniLM** as one of the embedding ablations (§7) so the paper can claim the result is not embedding-model-specific.

### 2.3 Text substrate

`description_core_llm` (with COALESCE fallback to raw `description` for the ~1% unlabeled rows) for both embedding and c-TF-IDF. This matches the project's substrate convention (`methodology_protocol.md` §1) and means c-TF-IDF top-words are not contaminated by EEO / benefits / recruiter boilerplate that would otherwise dominate cross-period comparisons.

### 2.4 Filters before sampling

- `description_core_llm IS NOT NULL`
- `LENGTH(description_core_llm) >= 200` (drops postings too short to embed reliably)
- `date_flag = 'ok'`
- `has_llm_classification = TRUE` (so we have L1/L2 labels for every clustered posting)
- `job_description_embedding IS NOT NULL` (~99.2% retention)

---

## 3. Sample

### 3.1 Three samples — exploration, confirmation A, confirmation B

The analysis splits the corpus into one exploration sample and two confirmation samples. The exploration sample is read first and only first; the confirmation samples are untouched until the L1/L2 taxonomy is frozen.

| Sample | Rows | Phase | Used for |
|---|---|---|---|
| **E — Exploration** | SWE postings from `kaggle_asaniczka` (2024-01) | 1 | Phase-1 BERTopic fit. Inputs to L1/L2 taxonomy revision. After freeze, this sample is **never** used to generate confirmation-set claims. |
| **A — Confirmation A (SWE)** | SWE postings from `kaggle_arshkon` (2024-04) + `scraped` (2026-03, 2026-04) | 2 | Phase-2 BERTopic headline fit. All paper claims about role-landscape change. |
| **B — Confirmation B (SWE + Control)** | Sample A plus control occupations from arshkon + scraped | 2 | Cross-occupation test. See §3.3 — fitted two ways and compared. |

#### Sampling cap — per (company × period × normalized title)

The cap prevents prolific firms from dominating cluster centroids. The right unit to cap on is **`company_name_canonical × period × title_normalized`**, not `company × period` alone. Reason: a firm posting 200 SWE roles split across 30 distinct titles is showing real role-mix breadth that we want to preserve, while a firm posting "Senior Software Engineer" 50 times in a week is showing literal duplication that we want to suppress. A pure company cap conflates these.

**Cap value: 5 per `(company_name_canonical, period, title_normalized)`.** Lower than the previous `30 per (company × period)` because the bucketing is finer — most buckets have far fewer than 5 members, so the cap binds only on actually-duplicated postings.

**Title normalization.** Lowercase, strip non-alphanumeric, collapse whitespace. This groups `Senior Software Engineer` / `senior software engineer` / `Senior  Software Engineer` → same bucket; keeps `Senior SWE III` and `Senior SWE II` as distinct (intentional — those are different roles). More aggressive normalization (e.g., synonymizing `Sr.` → `Senior`, removing roman-numeral seniority markers) is its own can of worms and not justified for a deduplication cap.

**Within-bucket selection.** `ROW_NUMBER() OVER (PARTITION BY canonical_co, period, title_normalized ORDER BY HASH(uid))`, take rn ≤ 5. The hash-on-uid order randomizes within bucket so we don't preferentially keep postings clustered in time (which would correlate with content for batch-posted reqs).

Aggregator inclusion is the default; aggregator exclusion is a robustness ablation (§7).

#### Sample sizes — to be confirmed in Stage 1

Pre-revision estimates (under the old `30 per (company × period)` cap) were ~14k for E, ~31k for A, ~61k for B. Under the new title-aware cap, sizes will shift — likely modestly upward, since title-bucketing makes most caps non-binding while only suppressing literal title-level duplication. Concrete sample sizes will be measured and recorded as the first output of Stage 1 (§12); this design doc commits to the rule, not the count.

Order-of-magnitude bounds: Sample A is between ~25k and ~50k SWE rows; the embedding tensor (3072-d float32) is between 0.3 GB and 0.6 GB. Comfortable in any case.

**Why asaniczka for exploration, not arshkon.** asaniczka is the larger 2024 source (19k SWE pre-cap vs 5k for arshkon). Larger n gives BERTopic more density signal for taxonomy development. Leaving arshkon in the confirmation set means Phase 2 still has a 2024 anchor for cross-period claims. The cost is that the cleanest within-2024 cross-source placebo is no longer available; we replace it with an exploration-vs-confirmation centroid alignment check (§6.6), which tests a stronger property — whether exploration-derived structure reproduces on completely unseen 2024 data.

### 3.2 Stratification record

Cluster outputs are reported with the following row-level breakdown alongside every cluster:

- n total, n by period, growth ratio (2026 / 2024)
- AI-vocab rate (broad regex, strict regex)
- Seniority distribution (junior / mid / senior shares; LLM abstention rate)
- Median YOE (where available)
- Top 5 firms; top 3 metros
- Aggregator share
- Within-cluster substrate-length distribution (sanity check vs boilerplate residue)
- **Sample provenance** (E / A / B) — every reported number in the paper carries this tag

This is the ground truth row that goes into the appendix datasheet — never less detailed than this.

### 3.3 Sample B — fit two ways, pick by reviewer agreement

There are two legitimate ways to bring control into the analysis:

- **B-joint.** Fit a fresh BERTopic on Sample B (SWE + Control). Produces one taxonomy that includes occupation-level structure (nurse, civil engineer, accountant clusters).
- **B-projected.** Fit BERTopic on Sample A only; project Sample B's control postings onto the existing model via `topic_model.transform(control_docs, control_embeddings)`. Produces topics-per-class on the SWE-derived structure.

B-joint may over-fragment SWE structure (the model spends capacity on cross-occupation distinctions); B-projected may force control postings into ill-fitting SWE clusters. Run both. The two-reviewer review (§5.2) labels clusters from each. The variant with stronger inter-reviewer agreement (and lower fraction of clusters flagged "incoherent") wins as the headline; the loser appears in the appendix as a sensitivity check.

---

## 4. Method

### 4.1 Pipeline

Standard BERTopic, with our pre-computed embeddings:

```
job_description_embedding (3072-d, OpenAI text-embedding-3-large)
   └─ UMAP (3072 → 5)              # cosine metric, seeded
       └─ HDBSCAN (min_cluster_size, min_samples)
           └─ c-TF-IDF on description_core_llm   # cluster representation
               └─ KeyBERTInspired              # representation refinement
                   └─ MaximalMarginalRelevance # n-gram diversification
                       └─ OpenAI GPT-5.5 / GPT-5.4-mini  # one-line cluster label (§5)
```

UMAP reduction is needed because HDBSCAN's distance computation is poorly behaved in 3072-d (curse-of-dimensionality, distance concentration). 5-d is the BERTopic default and matches our prior run; 10-d is an ablation.

### 4.2 Hyperparameters

Pre-registered. Frozen before fitting; ablations (§7) test sensitivity.

| Component | Param | Value | Why |
|---|---|---|---|
| UMAP | `n_neighbors` | 15 | BERTopic default; preserves local + some global structure |
| UMAP | `n_components` | 5 | BERTopic default; HDBSCAN-friendly |
| UMAP | `min_dist` | 0.0 | Encourage tight clusters for HDBSCAN |
| UMAP | `metric` | cosine | OpenAI embeddings are L2-normalized; cosine is the appropriate metric |
| UMAP | `random_state` | 42 (primary), 1337, 2026 (stability) | 3-seed protocol matches v3 prior |
| HDBSCAN | `min_cluster_size` | 60 (initial), tuned via §4.6 | At ~31k SWE rows in Sample A, 60 targets ~30–50 raw topics; finalized empirically |
| HDBSCAN | `min_samples` | 15 | Roughly 0.2× min_cluster_size; standard |
| HDBSCAN | `metric` | euclidean | Acts on UMAP output, not raw embeddings |
| HDBSCAN | `cluster_selection_method` | `eom` | Excess-of-Mass; matches prior |
| HDBSCAN | `prediction_data` | True | Needed for outlier reduction and approximate_distribution |
| CountVectorizer | `ngram_range` | (1, 3) | Single tokens + bigrams + trigrams capture multi-word skills ("vector database") |
| CountVectorizer | `min_df` | 10 | Drop hapaxes that c-TF-IDF will overweight |
| CountVectorizer | `max_df` | 0.4 | Drop terms in >40% of clusters (corpus-generic) |
| CountVectorizer | `stop_words` | English + custom | Custom list drops residual boilerplate tokens (see §4.5) |
| CountVectorizer | `token_pattern` | `(?u)\b[a-zA-Z][a-zA-Z\-\+/\.]+\b` | Keeps `c++`, `node.js`, `ci/cd`, `.net` |
| BERTopic | `min_topic_size` | match HDBSCAN `min_cluster_size` | Single source of truth |
| BERTopic | `calculate_probabilities` | False | Use `approximate_distribution` post-hoc instead — much cheaper |
| BERTopic | `nr_topics` | None initially; sweep K post-fit per §4.4 | K is characterized, not pre-fixed |

The `min_cluster_size` initial value (60) is a deliberate change from the v3 prior's 35. With denser OpenAI embeddings smaller values fragment more aggressively; 60 is chosen as the entry point of the sweep in §4.6 and finalized by criteria stated there.

### 4.3 BERTopic features we use

BERTopic's library API is large. The features we use, what each delivers, and where it appears in the paper:

| Feature | Purpose | Paper artifact |
|---|---|---|
| **Custom embeddings** (`embeddings=` kwarg in `fit_transform`) | Pass our precomputed OpenAI vectors directly; no in-pipeline encoding | All clustering |
| **Hierarchical topics** (`topic_model.hierarchical_topics(docs)`) | Build dendrogram of cluster mergers; pick a level for "super-family" view | Fig. 2 (super-family vs family resolution) |
| **Topics over time** (`topic_model.topics_over_time(docs, timestamps)`) | Per-period topic shares without re-fitting; uses static cluster assignment + per-period c-TF-IDF for vocabulary drift | Fig. 3 (drift words within stable clusters); Tab. 2 (growth/decline) |
| **Topics per class** (`topic_model.topics_per_class(docs, classes=...)`) | Topic shares conditioned on class (SWE-vs-control; junior-vs-senior; aggregator-vs-direct) without refitting | Fig. 4 (junior/senior topic split); §6 robustness |
| **Topic reduction** (`topic_model.reduce_topics(docs, nr_topics=K)`) | Merge similar topics to a target K | Sweep K per §4.4; headline K chosen by stability + interpretability criterion |
| **Outlier reduction** (`topic_model.reduce_outliers(docs, topics, strategy="distributions")`) | Reassign HDBSCAN noise points to nearest topic by soft-distribution | Reported as sensitivity: headline cluster shares **with and without** outlier reassignment |
| **Approximate distribution** (`topic_model.approximate_distribution(docs)`) | Soft topic membership per doc; needed for entropy claims | Concentration / entropy claims |
| **`find_topics(query, top_n=5)`** | Semantic search: pass a seed phrase ("AI agent orchestration"), get top topics by centroid similarity | Validation that pre-registered L1 families exist as data-driven topics |
| **OpenAI representation model** (`representation_model=OpenAI(client, model=<see §5.1.3>, chat=True)`) | LLM generates a one-line topic label from exemplar docs + top-words | §5 cluster naming |
| **`get_document_info(docs)`** | Per-doc: topic, prob, representative status | Per-row artifact `bertopic_assignments.parquet` |

Features we deliberately **do not** use:

- `online_topic_model` / `online_dim_reduction` — the corpus is static; no streaming need.
- `class_model.fit` (supervised topic modeling) — would defeat the discovery purpose.
- Built-in BERTopic visualizations (`visualize_topics`, etc.) — they ship as Plotly HTML; the paper requires `figures/style.py` (matplotlib + SciencePlots, Type 1/TrueType, AAAI/AIES sizing). We re-author each plot.
- `merge_models` / `merge_topics` based on multiple fits — risk of post-hoc taxonomy fishing.

**Honesty about "topic" terminology.** BERTopic does not perform probabilistic topic modeling in the LDA sense — it is a clustering wrapper over UMAP+HDBSCAN with c-TF-IDF for cluster labeling. Calling its outputs "topics" is convenient but blurs methodology. The methods subsection of the paper should write "data-driven cluster" or "role family cluster" rather than "topic" wherever the distinction matters. Reviewers familiar with LDA will appreciate the precision.

**Outlier-reduction strategy choice.** BERTopic offers four strategies (`c-tf-idf`, `embeddings`, `distributions`, `tokenset_similarity`). We use `embeddings` as the headline strategy — reassign each noise point to the topic whose centroid (in UMAP space) is nearest in cosine. Reasons: (a) consistent with the rest of the pipeline, which is geometry-driven; (b) `c-tf-idf` over-weights short postings whose few words dominate cosine vs the topic's TF-IDF vector; (c) `distributions` requires `calculate_probabilities=True`, which is much slower at our scale. We report cluster shares both with and without outlier reduction (§6.9); the strategy choice is one of the §7.2 ablations.

**Approximate-distribution parameters (for entropy / concentration claims).** When computing soft topic distributions per document via `topic_model.approximate_distribution(docs)`, pin: `window=8` (8-token sliding window), `stride=4` (50% overlap), `min_similarity=0.1` (drop near-zero affinity), `padding=False`. These match BERTopic defaults except `min_similarity` (which we set explicitly to drop noise from the entropy computation). Pinned in `config.py`.

### 4.4 Topic-count K — sweep, characterize, then commit

K is not a single pre-registered value. Different K answers different questions: small K shows the super-family structure (does an AI mega-region exist?); large K shows fine sub-archetypes (does AI-engineer split into RAG vs agents vs evals?). We characterize what changes across K rather than committing in advance to one resolution.

**Sweep grid (pre-registered):** K ∈ {10, 15, 20, 25, 30, 40, 50, 75}. Each K is generated by `reduce_topics(docs, nr_topics=K)` from the same raw fit, so cluster identities are nested.

**For each K, record:**

- Number of clusters that survive (some K may collapse below target if topics are very similar)
- Noise / outlier rate
- Mean inter-cluster centroid cosine
- Mean intra-cluster spread (cosine to centroid)
- DBCV score on UMAP output (if computable)
- Seed-pair ARI at this K (3 seeds, see §6.1)
- Per-period reproduction centroid alignment (§6.3)
- Coverage purity vs L1 (§6.7)
- Manual interpretability rating from a 5-cluster sample at this K (1–5 scale, both reviewers per §5.2)

**Headline K — selection criterion (pre-registered).** The headline K is the smallest K satisfying:

1. Seed-pair ARI ≥ 0.4
2. Per-period reproduction centroid alignment ≥ 0.85
3. Mean interpretability rating ≥ 3.5 / 5
4. Outlier rate ≤ 40%

If multiple K satisfy all four, pick the smallest (more parsimonious). If none does, headline K is reported as "no stable family resolution" and the paper makes only super-family-level claims.

**Two K reported in paper, regardless of headline pick:**

- A **super-family K** (somewhere in {10, 15}) for the broad map figure (F1).
- The **headline K** for per-cluster claims (T1).

The full K sweep table appears in the appendix (T4 ablation row), so reviewers can see how findings move across resolutions.

### 4.5 Custom stopwords

In addition to sklearn's English stopwords, drop:

- Substrate residue: `description`, `responsibilities`, `qualifications`, `requirements`, `position`, `role`, `team`, `work`, `looking`, `seeking`, `candidate`, `experience`
- Boilerplate that survived L9 stripping: `equal`, `opportunity`, `affirmative`, `disability`, `veteran`, `protected`, `sexual`, `gender`, `race`, `religion`, `applicant`, `employee`
- Generic recruiter-CTA: `apply`, `resume`, `interview`, `hiring`, `recruit`, `recruiting`
- Pure-noise tokens we observed in the v3 run: `etc`, `eg`, `ie`, `including`, `e.g`, `i.e`

This list is **frozen before fitting**. If we discover during analysis that another token dominates a cluster in an uninformative way, we document the finding — we do not silently grow the stopword list.

### 4.6 `min_cluster_size` sweep — to commit

Before the K sweep, we tune `min_cluster_size ∈ {30, 50, 60, 80, 100, 150}` on Sample A and record:

- Raw topic count (pre-reduction)
- Noise rate
- Mean inter-cluster cosine distance
- DBCV score (if computable on UMAP output)
- Post-`reduce_topics(30)` ARI vs neighboring sweep points

**Headline `min_cluster_size`** is the value at which (a) post-reduction ARI vs neighboring sweep points is ≥ 0.7 (stable plateau), AND (b) noise rate is in 15–35%. If two values qualify, pick the larger (more conservative). The chosen value is committed in `figures/bertopic/config.py` before §4.4's K sweep runs.

---

## 5. Cluster naming

This is the rigor-critical step. A cluster called "AI Engineer" instead of "Data + AI/ML hybrid" changes how every downstream claim reads. The naming protocol is:

### 5.1 Three independent name candidates per cluster

For each topic at the headline K (and additionally for clusters at the super-family K):

1. **c-TF-IDF top-15 terms.** Pure data-driven. Often unreadable as a label but disambiguating.
2. **KeyBERTInspired top-10 phrases.** Filters c-TF-IDF terms by similarity to the topic centroid embedding; tends to give better multi-word labels.
3. **OpenAI LLM label.** Prompt: "You are labeling a cluster of software-engineering job descriptions. Given these top words: {c-TF-IDF top-15} and these representative posting excerpts (titles + first 200 chars of description_core_llm): {5 exemplars}, produce a 2–4 word noun-phrase label that names the role family or sub-archetype. Do not invent vocabulary not present in the words or excerpts. Output JSON: {label, confidence: 0–1, alternative}."

**Model selection.** Use **GPT-5.5** when the total number of LLM calls in a single pass is < 100 (cluster naming at headline K + super-family K is ~30–50 calls — comfortably under). Switch to **GPT-5.4-mini** for any pass that exceeds 100 calls (e.g., naming across the full K sweep, or naming for both Sample A and Sample B variants in §3.3, or §6.10 cross-model robustness sweeps). Both models are pinned in `figures/bertopic/config.py`; switching mid-pass is forbidden.

**Exemplars** are selected as the 5 documents with highest `topic_model.get_representative_docs(topic_id)` (BERTopic-internal MMR over the cluster), supplemented with 2 random members so the prompt sees both prototypes and breadth.

### 5.2 Two-reviewer review — one author + one independent

The naming review is performed by **two reviewers**: one of the two authors (Jihyun or Gabor) and one independent reviewer recruited from outside the project (target: a senior SWE not involved in the data work; the same pool the qualitative-interview protocol draws from).

Each reviewer independently looks at: c-TF-IDF top-15 terms, 7 exemplar postings (5 representative + 2 random), and the 3 candidate names. Each writes:

- A preferred label (free-form, 2–6 words)
- A 1–5 confidence score
- A flag if cluster is incoherent (mixed semantics — should be split or merged)

The independent reviewer is given a one-page primer (the L1 family list, the four paper claims §1.2, the "naming should describe what the cluster *is*, not what we hope it is" instruction) but **not** told which clusters we expect to be emergent or shrinking — that would prime the labels.

Disagreements resolved by adjudication discussion (author drives; independent reviewer has veto on labels they consider misleading). Coherence flags resolved by re-running `reduce_topics` at one K-step finer with the flagged cluster's members excluded and seeing whether the cluster re-emerges (it should — if it doesn't, our naming was reflecting noise).

Inter-reviewer agreement is reported: % exact-match labels, mean cosine of label embeddings (sentence-transformer), and Cohen's κ on coherence flags. Disagreements are listed in the appendix, both labels shown.

### 5.3 Label-to-L1 mapping

After labels are committed, we cross-tabulate every confirmation-set BERTopic cluster against the (already-frozen) L1 family. Three outcomes per cluster:

- **One-to-one** with an L1 family (e.g., one BERTopic cluster cleanly maps to "Mobile (iOS/Android)") — confirms L1 families are in the held-out data
- **One-to-many** (one L1 family decomposes into multiple BERTopic sub-archetypes — e.g., "AI/LLM Engineer" → "RAG infrastructure" + "Agent orchestration" + "Foundation model research")
- **Unmapped** (BERTopic cluster fits no L1 family) — these are the candidate "emergent role" findings, OR signals that L1 was incomplete; we report and discuss but do **not** retroactively add to L1

Unmapped clusters get extra scrutiny: per-period growth, per-source check (does it appear in only scraped data, or in confirmation 2024 too?), per-firm check (is it dominated by one company that escaped the cap?).

### 5.4 Frozen-name commit

Once §5.1–5.3 are done, **labels are frozen**. They appear verbatim in every figure, table, and prose mention. Any later change to a label requires a paper-visible "Erratum: cluster X relabeled from Y to Z because …" note. This avoids the failure mode where labels drift to support claims.

---

## 6. Validation and rigor

For an AIES paper, we need to defend that the topic structure is real, not an artifact of seeds, embeddings, or sample. The defense is layered.

### 6.1 Stability — three seeds

Three seeds: 42, 1337, 2026. Same data, same hyperparameters, different UMAP `random_state`.

Report:

- **Topic-level ARI** between seed pairs (raw fits, before `reduce_topics`). Baseline: prior MiniLM run scored 0.49.
- **Family-level ARI** at headline K (and at K=30 specifically, for direct comparability with the v3 prior). Baseline: prior 0.44 at K=30.
- **Centroid alignment.** For each pair of seeds, find the best 1-to-1 cluster matching by Hungarian algorithm on centroid cosine, then report the mean centroid cosine of matched pairs. This catches the case where ARI is low because cluster boundaries shifted but centroids are stable — a much weaker structural failure than ARI suggests.

If any pair has ARI < 0.4 AND centroid alignment < 0.85, we say so plainly in the paper. We do not pick the friendliest seed.

### 6.2 Stability — bootstrap resampling

Three bootstrap samples of Sample A at 80% (without replacement, stratified by period and source). For each, refit BERTopic, then reduce to the headline K. Report ARI vs the headline fit on the overlapping rows. This addresses the "is the taxonomy a function of the specific 31k rows" worry.

### 6.3 Per-period reproduction

The prior v3 run used this and it was the load-bearing stability check. Refit BERTopic on **2024-only (arshkon)** and **2026-only (scraped)** subsets of Sample A (independent fits), reduce each to the headline K. For every joint-Sample-A cluster, find the nearest period-fit cluster by centroid cosine; report mean and median.

If joint-fit clusters reproduce on time-sliced data with mean centroid cosine ≥ 0.85, the cross-period comparison is trustworthy. The v3 baseline was 0.94/0.95.

This is a **stronger** claim than seed stability for our paper, because the temporal comparison is the whole point. We lean on it.

### 6.4 Cross-method comparison — NMF baseline

Fit NMF (k = headline K) on TF-IDF of `description_core_llm` (same vectorizer settings as c-TF-IDF). Hard-assign by `argmax`. Report ARI / NMI vs BERTopic. Different mathematical machinery → if they roughly agree, the structure is robust to method.

### 6.5 Cross-embedding comparison — MiniLM rerun

Refit BERTopic with `all-MiniLM-L6-v2` 384-d (the v3 prior's embedding) on Sample A. Reduce to the headline K. Compute ARI / NMI vs OpenAI fit. Two outcomes both fine for the paper:

- **Strong agreement (ARI ≥ 0.5).** Cluster structure is not embedding-specific.
- **Weak agreement (ARI < 0.3).** Cluster structure is embedding-specific — we name *which embedding* in every claim and are explicit that `text-embedding-3-large` reveals structure MiniLM does not.

### 6.6 Exploration ↔ confirmation reproduction (replaces the within-2024 placebo)

The original within-2024 placebo (asaniczka vs arshkon) is no longer clean: asaniczka has been used to inform L1/L2 (Phase 1). Instead, the analogous test is whether the **exploration-set BERTopic structure (Sample E) reproduces on the held-out confirmation 2024 slice (arshkon-only subset of Sample A)**.

Procedure:

1. Take the Phase-1 BERTopic model fitted on Sample E.
2. Restrict Sample A to its 2024 (arshkon) rows; embed with the same OpenAI model; project onto the Phase-1 model via `transform`.
3. Independently fit a fresh BERTopic on the arshkon-only 2024 confirmation slice; reduce to the headline K.
4. Compute centroid-alignment between Phase-1 (Sample E) and the arshkon-only fit using Hungarian matching on cluster centroids.

If the two 2024 fits agree (centroid cosine ≥ 0.85), the taxonomy generalizes within-year across sources and the cross-period (2024→2026) effects in Phase 2 are not source-confounded. If they disagree, the source-confound caveat is paper-visible.

This is now the strongest single test of generalizability in the analysis.

### 6.7 Convergent validity vs L1 — on the held-out confirmation set

After §5.3, we have L1-family-to-BERTopic-cluster crosstab on Sample A. **Critical:** this is the test that earns its name only because Sample A's data did not feed L1 development. Compute:

- **Mapping purity:** for each L1 family, what fraction of its postings live in its top-1 BERTopic cluster?
- **Mapping coverage:** for each BERTopic cluster, what fraction of its postings share the modal L1 family?

Targets: mean purity ≥ 0.6, mean coverage ≥ 0.7. Below these thresholds, the L1 taxonomy is not what's in the held-out data; we report and discuss, but do **not** tweak L1 — that would relitigate the freeze.

For comparison, also compute purity/coverage on Sample E (in-sample by construction). The gap between in-sample (E) and out-of-sample (A) purity/coverage is reported as the **L1 generalization gap** — a number reviewers will ask for.

### 6.8 Topic quality — coherence and diversity

Standard topic-modeling quality metrics, computed on c-TF-IDF top-10 terms per cluster, using Sample A as the reference corpus:

- **NPMI** (normalized pointwise mutual information). Range [-1, 1]; higher is better. Most-cited coherence metric in the LDA / BERTopic literature.
- **UMass** (log-conditional probability). Range (-∞, 0]; closer to 0 is better. Cheaper to compute, less correlated with human judgment.
- **C_v** (sliding-window cosine over normalized PMI). Range [0, 1]; the metric that correlates best with human topic-quality judgments per Röder et al. 2015. Compute via `gensim.models.CoherenceModel(coherence='c_v')`.
- **Topic diversity** (Dieng et al. 2020): fraction of unique tokens across all topics' top-10 lists. Low values flag taxonomies that are saying the same thing in different clusters.

Report mean, median, and distribution per metric. Targets are aspirational, not pre-registered cutoffs:

| Metric | "Reasonable" range |
|---|---|
| NPMI | mean ≥ 0.05 |
| C_v | mean ≥ 0.45 |
| Topic diversity | ≥ 0.6 |

These are not headline numbers but anchor reviewer questions about whether the clustering is sensible.

### 6.8b Cluster-size distribution and silhouette

Two more clustering-validity numbers, paper-appendix material:

- **Cluster size distribution** at the headline K. Report median, IQR, and the 5/95 percentiles in posts-per-cluster. Lopsided distributions (a single cluster with > 30% of postings) flag the v3 mega-cluster failure mode.
- **Silhouette score** in 5-D UMAP space (the same space HDBSCAN saw). Report mean per cluster and overall mean. v3 prior reported 0.27 inside the AI cohort — a known weak number; we expect comparable for new emergent regions and treat anything ≥ 0.4 as a strong-separation marker.

### 6.9 Honest noise rate

Report the HDBSCAN noise rate (-1 cluster) before and after `reduce_outliers`. Both numbers go in the paper. Do not silently use `reduce_outliers` to make the noise rate look smaller — that strategy reassigns ambiguous postings to whichever cluster is geometrically nearest, which may not reflect their content.

### 6.10 Cross-model cluster-name robustness

Re-run §5.1.3 with `claude-opus-4-7` as a second labeler (this is a per-cluster pass — for headline K it's < 100 calls and runs at the same model tier as the OpenAI namer). Compute exact-match label rate and a 5-point semantic-equivalence score (rated by both reviewers from §5.2). Disagreements are surfaced in the appendix; if either model produces a label that materially changes how a cluster reads, that's a paper-visible flag.

---

## 7. Ablations and sensitivity

Two tiers: **primary characterization** (K and `min_cluster_size` — both swept and reported in the body) and **secondary ablations** (single perturbations, appendix only). Each ablation reports ARI vs headline plus a qualitative description of the largest cluster movement.

### 7.1 Primary characterization (in-body)

| Parameter | Grid | Reported as |
|---|---|---|
| **Topic count K** | {10, 15, 20, 25, 30, 40, 50, 75} | Sweep curve: per-K stability ARI, interpretability rating, L1 purity (§4.4); body figure |
| **HDBSCAN `min_cluster_size`** | {30, 50, 60, 80, 100, 150} | Sweep table per §4.6; body if headline value is contentious, appendix otherwise |

### 7.2 Secondary ablations (appendix Table A4)

Single perturbations against the headline fit. Reported as ARI vs headline + qualitative description of the largest cluster movement.

| Ablation | Variant | Reason |
|---|---|---|
| **Embedding model** | MiniLM-L6, jobBERT-v2, OpenAI text-embedding-3-small | Embedding-specificity (also §6.5) |
| **UMAP `n_components`** | 5 (headline), 10, 15 | Compression sensitivity |
| **UMAP `n_neighbors`** | 15 (headline), 30, 50 | Local-vs-global structure |
| **Sample cap** | 5/(co × period × title) (headline), 3 and 10 per same bucket, plus the legacy 30/(co × period), plus uncapped | Prolific-employer distortion; title-aware vs title-blind |
| **Aggregator** | Include (headline) vs exclude | Aggregator-mediated content |
| **Substrate** | `description_core_llm` (headline) vs raw `description` | Boilerplate residue |
| **Length floor** | 200 (headline), 100, 400 | Short-posting noise |
| **Outlier reduction** | Off (headline), on (`distributions` strategy) | Soft-vs-hard cluster shares |
| **Sample B variant** | B-projected (headline if it wins §3.3), B-joint (else) | Cross-occupation strategy |
| **Exploration set choice** | asaniczka (headline) vs random 30% slice of full corpus | Did the asaniczka-as-exploration choice bias L1? |

Ablations live in appendix Table A4. We commit to publishing the table whether or not it's friendly.

---

## 8. Deliverables

### 8.1 Artifacts (machine-readable)

| Path | Contents |
|---|---|
| `data/bertopic/exploration/model.bertopic` | Phase-1 fitted model on Sample E. Frozen and never refitted post-L1 freeze. |
| `data/bertopic/exploration/cluster_to_l1_input.csv` | Phase-1 clusters → proposed L1 family additions/merges/splits, with rationale per cluster. Becomes the audit trail for the L1 freeze. |
| `data/bertopic/confirmation/assignments.parquet` | (uid, topic_id, topic_label, prob, is_outlier, sample) — primary integration artifact, joined back to `unified_core.parquet` for downstream figures. Sample E rows tagged `sample=E` and excluded from confirmation-set claim tables. |
| `data/bertopic/confirmation/topic_info.parquet` | One row per topic: id, label, n, n_2024, n_2026, growth, c_tf_idf_top, keybert_top, llm_label, llm_alt, reviewer_author_label, reviewer_independent_label, l1_modal, l1_purity |
| `data/bertopic/confirmation/topics_over_time.parquet` | (topic_id, period, n, c_tf_idf_top_5) — for vocabulary-drift figure |
| `data/bertopic/confirmation/topics_per_class.parquet` | (topic_id, class_var, class_value, n) — for SWE-vs-control and junior-vs-senior splits |
| `data/bertopic/confirmation/hierarchy.parquet` | Dendrogram from `hierarchical_topics` — for super-family figure |
| `data/bertopic/k_sweep.parquet` | One row per K in §4.4 grid: K, n_clusters, noise_rate, seed_ari, period_alignment, l1_purity, interp_rating |
| `data/bertopic/stability.parquet` | Per-pair ARI / NMI / centroid-alignment for §6.1, §6.2, §6.3, §6.6 |
| `data/bertopic/ablations.parquet` | One row per §7 ablation: name, ARI vs headline, noise rate, n topics |
| `data/bertopic/embeddings_cache.npy` | Cache of OpenAI embeddings used (dedup with unified_core but explicit for the run) |

### 8.2 Figures (paper, all via `figures/style.py`)

- **F1 — UMAP map.** 2×2 small multiples: rows (period: 2024 / 2026) × columns (sample: SWE / SWE+Control). Points colored by BERTopic family at headline K. Sized for `FIGSIZE_DOUBLE`.
- **F2 — Hierarchy.** Dendrogram of headline-K clusters merging up to the super-family K. Labels frozen per §5.4.
- **F3 — Growth/decline.** Diverging horizontal bars: per-cluster Δ posting share (2026 − 2024), sorted by absolute. Color-coded as growers / shrinkers / stable.
- **F4 — Vocabulary drift within stable clusters.** 4 typographic call-outs (no axes) for the 4 clusters with cleanest within-cluster c-TF-IDF rotation: `entered` (2026 ∖ 2024) and `exited` (2024 ∖ 2026) terms. Pre-registered as the v3 prior's panel D pattern.
- **F5 — Concentration.** Two panels: (left) Lorenz curve of cluster-share by period; (right) effective number of clusters (entropy-based) over the four period bins. Exploration-vs-confirmation centroid-alignment overlaid as the within-year reference.
- **F6 — Topics per seniority.** Heatmap: cluster × {junior, mid, senior}, cell values = posting share within seniority class.
- **F7 — K sweep.** Body figure characterizing how stability ARI, L1 purity, and noise rate move across K ∈ {10, 15, 20, 25, 30, 40, 50, 75}. Justifies the headline K visually rather than asking readers to trust it.

### 8.3 Tables

- **T1 — Cluster catalog.** Rows: headline-K clusters. Cols: label, top-words, n_2024, n_2026, growth, AI-vocab rate, median YOE, top firms, top metros, L1 family modal, mapping purity.
- **T2 — Stability.** Seed pairs, bootstrap pairs, per-period reproduction, exploration↔confirmation alignment — all centroid-alignment and ARI numbers in one table.
- **T3 — Convergent validity.** ARI / NMI: BERTopic-vs-NMF, OpenAI-vs-MiniLM, joint-vs-2024-only, joint-vs-2026-only, exploration-vs-confirmation, in-sample-vs-out-of-sample L1 generalization gap.
- **T4 — Ablations (full).** §7.2 secondary ablations, exhaustive. Appendix.
- **T5 — Naming concordance.** Per cluster: c-TF-IDF top-3, OpenAI label, Claude label, author-reviewer label, independent-reviewer label, agreement flags.
- **T6 — Robustness landscape.** §8.5 sign-consistency matrix: rows = headline + each §7.2 ablation, cols = primary claims C1–C4, cells = ✓ / ✗ / partial / N/A. **In-body**, the only ablation summary that reaches the body of the paper.

### 8.4 Prose deliverables

- A 1–2 page methods subsection in the paper, drawn from §1–§6 of this doc. Same caveats, half the words. Must call out the exploration/confirmation split explicitly.
- Two appendix sections: pre-registration log (§10), full ablation table (§7).

### 8.5 Presenting ablations in the paper

For an AIES paper, ablations should be **doubly visible**: a small in-body summary that reviewers can read in 30 seconds, plus the full table in the appendix. The standard pattern in posting-research and clustering papers is three artifacts working together:

**1. The ablation table (T4) — exhaustive, appendix.** One row per perturbation. Columns: name, variant value, n clusters, ARI vs headline, mean centroid alignment vs headline, noise rate, and one column per primary claim (C1–C4 from §1.4) showing whether the claim still holds (✓ / ✗ / partial / N/A). Reviewers consult this when they have a specific concern.

**2. The robustness landscape (T6) — sign-consistency, in-body.** A compact matrix:

| | C1 (AI cluster emerged) | C2 (legacy shrunk) | C3 (rewriting) | C4 (concentration) |
|---|---|---|---|---|
| Headline | ✓ | ✓ | ✓ | ✓ |
| MiniLM embedding | ✓ | ✓ | ✓ | ✓ |
| `min_cluster_size` = 30 | ✓ | ✓ | partial | ✓ |
| Aggregator excluded | ✓ | ✓ | ✓ | partial |
| Substrate = raw | ✓ | ✓ | ✓ | ✓ |
| Per-firm cap = 10 | ✓ | ✓ | ✓ | ✓ |
| Per-firm cap = uncapped | ✓ | partial | ✓ | partial |
| K = 20 | ✓ | ✓ | ✓ | ✓ |
| K = 50 | ✓ | partial | ✓ | ✓ |
| ... | | | | |
| **Sign-consistency** | **N/N ✓** | **(N-2)/N ✓ + 2 partial** | **(N-1)/N ✓** | **(N-2)/N ✓ + 2 partial** |

The bottom-row summary is the headline number reviewers will quote: "claim C1 holds across all N robustness checks; C2 and C4 hold cleanly across all but two." This is the version of "the result is robust" that earns its name. We pre-commit to **publishing this table whether or not it's friendly** — partial / failing cells are not redacted.

The sign-consistency convention: ✓ = effect is in the same direction at p < 0.05 with effect size within ±30% of headline; ✗ = effect reverses or vanishes; partial = effect same-direction but effect size outside ±30%.

**3. Sensitivity bands on figures.** For figures reporting a primary 2024→2026 effect (F3 growth/decline bars; F5 concentration), shade the range across §7.2 ablations as a band around each bar / point. Readers see at a glance whether the effect is robust without consulting T6.

**Code release.** AIES expects a public repository. We release: the reproducible Stage 3 notebook (§12), `figures/bertopic/` source, the prereg log, the embeddings cache (anonymized — uid-keyed, no PII), and the cluster assignments parquet. The exploration/confirmation split is preserved in the release directory structure, so a re-runner can verify the claim-of-non-circularity by inspection.

**What we do not do.** We do not show cherry-picked ablations as "robustness" while silently dropping unfriendly ones; we do not present only the median across ablations as a single number without showing the range; we do not relegate failing ablations to a footnote.

---

## 9. Risks and known limits

### 9.1 The mega-cluster problem may persist

The v3 prior found that AI/ML and traditional data-science postings collapsed into one density region. OpenAI 3072-d embeddings are denser and may resolve this — or may not. If the mega-cluster persists:

- We do **not** quietly split it with the AI-vocab regex and present the split as a found taxonomy. The v3 prior was explicit that this was a post-hoc split; we stay explicit.
- We try a hierarchical sub-clustering on the mega-cluster alone (re-run UMAP + HDBSCAN on its members) before resorting to vocab split.
- If neither resolves it, we present the mega-cluster as one family and characterize its internal vocabulary shift over time as the key finding — the v3 prior framing.

### 9.2 c-TF-IDF cross-period vocabulary turnover

If new vocabulary enters the corpus in 2026 (e.g., "agentic", "RAG"), c-TF-IDF will up-weight those terms simply because they're rare elsewhere. This artificially makes 2026-period cluster representations look more "AI-flavored" than is warranted by the actual posting shift. Mitigation:

- Compute c-TF-IDF on the **pooled** corpus (default), not per-period
- Use `topics_over_time` for per-period vocabulary diff, where the prior IDF is from the joint corpus
- Report top-words alongside per-period vocabulary diff so readers can distinguish

### 9.3 Domain-vertical vs technical-horizontal axes confound

The v3 prior's healthcare/biotech AI cluster was a **vertical** specialization that the embedding split because the domain vocabulary dominated. We expect a similar healthcare/biotech island and possibly a fintech island. These are not "new role families" — they are domain verticals of existing role families. The labeling protocol (§5) and L1 mapping (§5.3) must catch this; the prose must call it out.

### 9.4 LLM-naming priming

If the OpenAI naming prompt sees exemplars dominated by AI vocabulary, it labels the cluster "AI Engineer" even when only 60% of postings are AI-flavored. Mitigation:

- Naming model sees both top exemplars and 5 random members (BERTopic's `get_representative_docs` is MMR-like, but supplement with random)
- Two-reviewer review (§5.2) is the actual safety net — if either reviewer disagrees with the LLM label, the LLM label loses
- Alternative-name field forces the LLM to consider a second framing

### 9.5 Seed-sensitive fine structure

If §6.1 ARI < 0.4 at every K in the §4.4 sweep, individual cluster identities are not stable across seeds and per-cluster claims are weak. In that case:

- Drop to super-family granularity (the hierarchy from §4.3)
- Report only super-family-level claims (8–10 super-families)
- Per-cluster claims become illustrative not load-bearing

We pre-commit that K is set by stability, not by our preference.

### 9.6 OpenAI embedding deprecation / drift

`text-embedding-3-large` is the current model. If OpenAI silently changes the model (they've done this), our results become non-reproducible. Mitigation: cache embeddings to `data/bertopic/embeddings_cache.npy` keyed by uid; never re-fetch unless the uid changes. The repo has this convention via Stage 11; we extend it.

### 9.7 22-month observation gap — not BERTopic-specific

The paper's broader 22-month gap between 2024 and 2026 affects all cross-period claims, including BERTopic. We do not solve it here; we cite the paper's gap discussion and confine BERTopic claims to "describes posting-frame change," not "describes employment change."

---

## 10. Pre-registration

**Frozen before fitting.** This is what we commit to.

### 10.1 Hyperparameters

§4.2 table. No changes once fit begins, except via §4.6's sweep protocol.

### 10.2 Sample definition and exploration/confirmation discipline

§3.1 — Sample E (asaniczka 2024 only) for Phase 1; Sample A and Sample B for Phase 2. Filter rules in §2.4. Sample cap: 5 per `(company_name_canonical × period × title_normalized)`, hash-randomized within bucket.

**Exploration/confirmation discipline (load-bearing):**

- Phase 1 BERTopic on Sample E only.
- L1 and L2 frozen at the end of Phase 1, with all changes vs the literature draft documented in `data/bertopic/exploration/cluster_to_l1_input.csv`.
- After freeze, Sample A and Sample B may be touched. Sample E results are not refit, retuned, or reread to influence Phase 2 hyperparameter choice except via this doc.
- Discoveries about L1 incompleteness on Sample A are *reported as findings*, not used to update L1.

### 10.3 K — sweep, not single value

K is **swept** across {10, 15, 20, 25, 30, 40, 50, 75} and characterized per §4.4. The headline K is selected by the four pre-registered criteria in §4.4 (seed ARI ≥ 0.4, period reproduction ≥ 0.85, interpretability ≥ 3.5/5, outlier ≤ 40%). The full sweep is reported in F7 + appendix; the headline K and one super-family K go in the body.

### 10.4 Seeds

42 (primary), 1337, 2026 (stability). No silent reseeding.

### 10.5 Stopwords

§4.5 list. Frozen pre-Phase-1. No additions during analysis.

### 10.6 Cluster naming protocol

§5.1–§5.4. **Models pinned in `config.py`:** GPT-5.5 for passes < 100 calls, GPT-5.4-mini for passes ≥ 100 calls; no mid-pass switching. Two reviewers per §5.2 — one author, one independent. Labels frozen post-§5.4; any later change requires a paper-visible erratum note.

### 10.7 Validation thresholds — stated in advance

| Check | Threshold | If violated |
|---|---|---|
| Per-period reproduction centroid cosine | ≥ 0.85 | Cross-period claims demoted to within-cluster only |
| Seed-pair ARI at headline K | ≥ 0.4 | Per-cluster claims demoted; super-family-level only |
| Exploration ↔ confirmation alignment (§6.6) | ≥ 0.85 centroid cosine | Cross-period effects flagged as partly source-confounded |
| L1 mapping purity (mean, on Sample A) | ≥ 0.6 | L1 taxonomy not what's in held-out data; reported, not adjusted |
| L1 generalization gap (E vs A purity) | ≤ 0.20 | L1 over-fit to exploration set; reported |
| Outlier rate (HDBSCAN) | ≤ 40% | If higher, retune `min_cluster_size` once via §4.6, document the change |

### 10.8 Things we will not do

- Hand-tune UMAP / HDBSCAN to make a specific cluster appear or disappear
- Choose headline K to maximize a specific finding's effect size (selected by §4.4 criteria, full)
- Drop seeds whose ARI is inconvenient
- Rename clusters to support claims after labels are frozen
- Use `reduce_outliers` and report the post-reduction noise rate as the only number
- Add stopwords mid-analysis to make a cluster's top-words read more cleanly
- Re-run with a different embedding model and silently switch headlines
- Refit Sample E with new hyperparameters after Sample A has been read
- Update L1 / L2 in response to Sample A findings

### 10.9 Paper-visible audit trail

A pre-registration log (`data/bertopic/prereg_log.md`) records: timestamp of this doc's commit, Phase 1 freeze timestamp (with the diff between the literature draft of L1/L2 and the post-Phase-1 frozen version), Phase 2 run timestamps, any deviation from this doc with rationale, and the final hyperparameter values. The log goes in the paper appendix.

---

## 11. Compute and runtime estimate

| Step | Time (rough) | Memory |
|---|---|---|
| **Phase 1 — Sample E (~14k × 3072)** | | |
| Load embeddings | < 30 s | 0.17 GB |
| UMAP + HDBSCAN | 2–4 min | 1 GB |
| c-TF-IDF + KeyBERTInspired + LLM naming (~30 calls @ GPT-5.5) | 2–3 min, ~$1 | 0.5 GB |
| L1/L2 review work (manual) | hours, not minutes | trivial |
| **Phase 2 — Sample A (~31k × 3072)** | | |
| Load embeddings | < 30 s | 0.36 GB |
| UMAP fit (3072 → 5) | 3–6 min on CPU; < 2 min with `n_jobs=-1` | 1–2 GB |
| HDBSCAN | 30 s – 2 min | 0.5 GB |
| K sweep (8 values via `reduce_topics`) | 1–2 min total | 0.5 GB |
| LLM naming (headline K + super-family K, ~50 calls @ GPT-5.5) | 2–3 min, ~$2 | trivial |
| Stability (3 seeds, full repeat) | ~3× above | same |
| Sample B-projected and B-joint variants | ~5 min + ~10 min | up to 1.5 GB |
| §6 validation suite (per-period, NMF, MiniLM, exploration↔confirmation) | ~30 min total | up to 3 GB |
| §7.2 secondary ablations (~10 reruns) | ~2–3 hr total | up to 3 GB |
| **End-to-end with caching** | **Phase 1 ~ 1 hr; Phase 2 ~ 4 hr first time, < 30 min on rerun** | **peak ~4 GB** |

Well within the 31 GB constraint. Caches: UMAP output, HDBSCAN output, c-TF-IDF matrix, LLM labels — all keyed on the sample's uid hash, separated by sample (E / A / B).

---

## 12. Implementation plan

The work has three stages aligned to the project's broader research workflow. **Stage 1** is exploratory — Claude-managed scripts, fast iteration, broad parameter coverage. **Stage 2** is the decision gate — two-person review of Stage 1 outputs to commit hyperparameters, ablations, figures, and the L1/L2 freeze. **Stage 3** is the consolidation — a single reproducible IPython notebook that anyone can re-run end-to-end to produce the paper's BERTopic figures and tables.

Within Stages 1 and 3 the Phase 1 → freeze gate → Phase 2 logic still applies — both phases get exploratory script coverage in Stage 1 and reproducible-notebook coverage in Stage 3.

### Stage 1 — Exploratory scripts (Claude-managed, fast iteration)

Code lives in `figures/bertopic/` as standalone Python scripts. Output is intermediate — CSVs, scratch parquets, debug PNGs in `figures/bertopic/intermediate/`. Claude runs scripts, summarizes findings in short memos in `figures/bertopic/memos/`.

**Stage 1a — Phase 1 exploration on Sample E:**

S1. **Literature draft of L1 and L2** — author work, no code. `literature_draft_l1_l2.md` with citations. *(Authors)*
S2. **Sampling validation script** — confirm the new title-aware cap (§3.1) gives the expected sample sizes; flag any company-period buckets dominating. Output: `intermediate/sample_E_size.csv`, `intermediate/sample_A_size.csv`. *(Claude script)*
S3. **`min_cluster_size` sweep on Sample E** — §4.6 protocol. *(Claude script)*
S4. **Phase-1 BERTopic fit + K sweep on Sample E** — §4.4 protocol. *(Claude script)*
S5. **Lightweight LLM naming on Sample E clusters** — §5.1.3 model selection. Naming for taxonomy purposes only, not paper labels. *(Claude script + author review)*
S6. **L1/L2 revision drafting** — walk the literature draft against the Phase-1 cluster list. Output: `data/bertopic/exploration/cluster_to_l1_input.csv` with rationale per change. *(Authors, Claude assists)*

**Stage 1b — Phase 2 exploration on Sample A and B (after freeze):**

Begins only after Stage 2's L1/L2 freeze.

S7. **Headline fit on Sample A + K sweep** — §4.4. *(Claude script)*
S8. **Stability runs** — seeds, bootstrap, per-period reproduction (§6.1–6.3). *(Claude script)*
S9. **Cross-method/embedding/exploration↔confirmation** — §6.4–6.6. *(Claude script)*
S10. **Topic-quality metrics** — coherence + diversity + silhouette (§6.8 / §6.8b). *(Claude script)*
S11. **Sample B — both variants** — B-projected and B-joint (§3.3). *(Claude script)*
S12. **Full §7.2 secondary ablations sweep.** *(Claude script)*
S13. **Cluster naming + reviewer review** — §5 full protocol. *(Claude prepares packets, reviewers rate)*

Each script run produces a short memo: what was run, what was found, what looks wrong, what should change before Stage 3. Memos are the input to Stage 2.

### Stage 2 — Decision gate (two-person review)

Run once between Stage 1a and Stage 1b (the L1/L2 freeze), and once between Stage 1b and Stage 3 (the figure / claim freeze).

**Stage 2a — L1/L2 freeze (after S6):**

- Both authors review S2–S6 memos
- Sign off `cluster_to_l1_input.csv`
- Update `prereg_log.md` with: literature-draft hash, frozen-L1 diff vs draft, freeze timestamp
- L1/L2 LLM application kicks off in the upstream preprocessing pipeline (blocking for S9's §6.7)

**Stage 2b — Figure / claim / ablation freeze (after S7–S13):**

- Review which §7.2 ablations actually moved any §1.4 claim, drop the dead ones from T4 / T6 (with explicit rationale in the prereg log)
- Review the K-sweep curve, commit headline K per §4.4 criteria, update `config.py`
- Review naming concordance, commit frozen labels per §5.4
- Decide which §1.4 claims survive: which C/T rows fired and which falsified, paper narrative restructured to match
- Pick Sample B variant per §3.3
- Pick figures: which of F1–F7 actually carry their weight; cut the rest

After Stage 2b, the headline result set is frozen. Stage 3 is consolidation, not re-exploration.

### Stage 3 — Reproducible notebook (publishable artifact)

A single IPython notebook, `figures/bertopic/bertopic_paper.ipynb`, that:

- Imports from `figures/bertopic/config.py` (frozen hyperparameters, LLM models, sample-cap rules)
- Loads `data/unified_core.parquet` and `data/bertopic/embeddings_cache.npy`
- Builds Sample E, A, B per the frozen rules
- Refits Phase 1 and Phase 2 models from cached embeddings (no re-encoding)
- Reads the frozen `cluster_to_l1_input.csv` and frozen cluster labels
- Reproduces every figure (F1–F7) and table (T1–T6) with identical numbers to the paper
- Runs end-to-end in < 30 minutes on the cached path; clearly flags which cells are slow / require API access

The notebook is the artifact released alongside the paper. Anyone with the data and a frozen `config.py` can re-run it. The exploratory scripts in Stage 1 stay in the repo for audit but are not the headline artifact.

**If Stage 1 finds something that demands re-exploration**, that's a Stage 2 decision — formally reopen, document in the prereg log, re-enter Stage 1 with explicit rationale. We do not silently rerun Stage 1 after Stage 2 sign-off.

### 12.1 Code structure

```
figures/bertopic/
  __init__.py
  config.py                       # frozen hyperparameters, LLM model pinning per §5.1.3, sample-cap rules
  sample.py                       # build Sample E, Sample A, Sample B (DuckDB + pyarrow)
  literature_draft_l1_l2.md       # S1 literature draft
  exploration/
    fit.py                        # S4 Phase-1 fit
    naming.py                     # S5 lightweight naming
    revise_l1_l2.py               # S6 produces cluster_to_l1_input.csv
  confirmation/
    fit.py                        # S7 Phase-2 fit; B-projected and B-joint variants (S11)
    naming.py                     # S13 §5 full protocol
    validation.py                 # S8–S10 §6 suite
    sample_b.py                   # S11 §3.3 two-variant comparison
  ablations.py                    # S12 §7.2
  artifacts.py                    # write all parquets in §8.1
  intermediate/                   # Stage-1 scratch outputs (gitignored)
  memos/                          # Stage-1 short memos (committed)
  fig_*.py                        # one file per F1–F7
  tab_*.py                        # one file per T1–T6
  bertopic_paper.ipynb            # Stage-3 reproducible notebook (the release artifact)
prereg_log.md                     # §10.9 audit trail, copied to paper appendix
```

`config.py` is the single source of truth for hyperparameters, LLM model pins, and sample-cap rules; everything else imports from it. The `exploration/` ↔ `confirmation/` directory split mechanically enforces the freeze gate: no `confirmation/*.py` file imports from `exploration/*.py` except via the frozen `cluster_to_l1_input.csv`. The reproducible notebook (`bertopic_paper.ipynb`) imports from both but does not redefine any hyperparameter.

### 12.2 Code style and quality bar

Mandatory for every script and notebook in `figures/bertopic/`. Apply Clean Code (Martin) plus post-Martin refinements (Tidy First, Rule of Three, YAGNI, Ousterhout on deep modules). The bar: minimum mass for the job, every line earning its place. Future agents must pick the codebase up cold without surprises.

- **Naming.** Intent-revealing. Verbs for functions (`fit_phase1`, `compute_purity`); nouns for modules (`sample`, `validation`). Booleans are predicates (`is_outlier`, `has_label`). No abbreviations except {`df`, `n`, `k`, `ax`, `fig`, `np`, `pd`}.
- **Functions.** Do one thing — if "and" appears in the description, split. ~30 lines max. ≤ 3 arguments where possible (else use a dataclass). Pure where practical. Return types annotated.
- **Comments.** Default: write none. Write one only when the *why* is non-obvious — a hidden constraint, an empirical finding, a workaround. Never restate the code; never leave commented-out code; no change-history comments (git log is authoritative). Module docstring: one paragraph, what + why.
- **Files.** One responsibility per file. No `utils.py` / `helpers.py` — name what the file does. Delete unused files in the same commit. The repo is not a journal: no abandoned drafts, no scratch experiments past their use, no `_old` / `_v2` suffixes. Only `bertopic_paper.ipynb` is committed as a notebook.
- **Abstraction discipline.** YAGNI — no speculative code for hypothetical futures. Rule of Three — no helper until three call sites exist; two similar blocks is fine. Don't wrap library calls that don't need wrapping (`BERTopic(...)` direct beats a forwarder named `make_topic_model`). No dynamic imports, no metaclasses.
- **Configuration.** Every hyperparameter in `config.py`. No magic numbers in scripts. Every seed enumerated. All paths derived from `config.PROJECT_ROOT` — no hard-coded absolute paths, no `~/`.
- **Determinism.** Caches content-addressed by sample-hash + config-hash. Same `config.py` produces byte-identical outputs. If a re-run drifts, fix the drift before the next commit.
- **Failure mode.** Fail fast and loud. No silent `except: pass`. Validate at boundaries (parquet read, OpenAI response, file IO); trust your own functions internally.
- **Tidy First (Beck).** Separate refactor commits from behavior-change commits. A reviewer should read each kind independently.
- **Tooling.** `ruff` formats and lints; the repo's config is the only opinion. `mypy --strict` clean on `figures/bertopic/` before Stage 2 sign-off.
- **Boy Scout.** Leave nearby code cleaner than you found it. If you understand a messy bit while passing through, fix it in the same commit.

If a rule above conflicts with the existing repo's conventions, the repo wins — consistency over local optimization.

---

## 13. Reproducibility and code release

AIES expects an accessible reproduction path. Our release plan:

| Artifact | Released | Location |
|---|---|---|
| Stage 3 notebook (`bertopic_paper.ipynb`) | Yes | Public repo |
| `figures/bertopic/` source (Stage 1 scripts + Stage 3 helpers) | Yes | Public repo |
| `config.py` (frozen hyperparameters, LLM model versions) | Yes | Public repo |
| Pre-registration log (`prereg_log.md`) | Yes | Public repo |
| Cluster assignments (`assignments.parquet`) | Yes | Public repo |
| Cluster labels and naming-concordance (T5) | Yes | Public repo |
| OpenAI embedding cache, uid-keyed | Yes — large file via Git LFS or a separate release tarball | Public repo / release |
| `unified_core.parquet` slice for SWE/control | Yes — but only fields used by the notebook (uid, source, period, title, description_core_llm, embedding, is_swe, is_control, seniority_final, company_name_canonical, etc.) | Public repo |
| Raw scraped HTML / pre-pipeline data | No — covered by paper's data-availability section, not the BERTopic repo |
| LLM-naming raw API responses | Yes, sanitized | Public repo |
| Reviewer label sheets (T5 source) | Yes, anonymized | Public repo |

**Pinned dependencies.** `requirements.txt` for the BERTopic release pins exact versions of: `bertopic`, `umap-learn`, `hdbscan`, `sentence-transformers`, `scikit-learn`, `gensim`, `openai`, `anthropic`, `pyarrow`, `duckdb`, `numpy`, `scipy`. A `Dockerfile` is nice-to-have, not required. We test the notebook end-to-end on a clean venv before submission.

**Embedding-model versioning.** OpenAI embeddings are not deterministic across silent server-side updates (§9.6). The cache is the canonical artifact: any re-encoding produces a number that may differ from the paper's. `config.py` records the exact OpenAI model string and the date of the cache build; `prereg_log.md` records the cache hash.

**Ethics statement (BERTopic-specific).** Two concerns and our mitigations:

- **LLM cluster-name bias.** GPT-5.5/5.4-mini may name clusters using stereotyped or marketing-flavored vocabulary inherited from training data. Two-reviewer override (§5.2) is the safety net. Reviewer-amended labels appear in T5 alongside the LLM proposal; readers see the diff.
- **Public-data posture.** The cluster assignments are derived from public LinkedIn job postings. We release uid-keyed assignments without re-publishing posting text in the release tarball — anyone who wants posting text fetches it via the upstream pipeline against their own access to the source datasets, per the paper's broader data-availability statement.

**End of design doc.**
