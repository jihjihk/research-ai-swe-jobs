# BERTopic discovery and embedding-space analysis — design doc

**Status:** draft, pre-registration target. Owner: Jihyun + Gabor. Conference target: AIES 2026.

This document specifies the BERTopic and embedding-space analyses that underpin the paper's "role landscape" claims (role families emerging, legacy stacks shrinking, within-family vocabulary drift, role concentration). The work is a discovery instrument: it tells us what density regions exist in the SWE posting corpus, what their vocabulary is, how those regions move between 2024 and 2026, and where the embedding space encodes structure that BERTopic alone misses. L1 (role family) and L2 (skill theme) classifications are produced separately by another agent against pre-committed taxonomies; this plan does not build, validate, or depend on them — it cross-references them in a strict read-only way once they exist (§7.7).

The doc is split into fourteen sections: what this analysis is doing in this paper (§1), inputs (§2), sample (§3), BERTopic method (§4), cluster naming (§5), embedding-space analyses (§6), validation (§7), ablations (§8), deliverables (§9), risks (§10), pre-registration (§11), runtime (§12), implementation plan and sub-agent delegation (§13), reproducibility and code release (§14). The most decision-critical section to read first is §1.4 — what claims this analysis can prove or disprove. The most operationally critical section is §13 — how the work is staged and delegated.

---

## 1. Purpose and scope

### 1.1 Where this analysis sits in the methodology

The paper has three classification layers; this plan owns one of them and uses the others read-only.

| Layer | Purpose | Method | Owned by |
|---|---|---|---|
| **L1 — Role family** | Assign each posting to one of ~17 role families (Frontend Web, Backend API, AI/LLM Engineer, …) | One LLM call per posting against a frozen pre-committed prompt | **Separate agent**, not this plan |
| **L2 — Skill themes** | Tag each posting on 8 work themes (orchestration, verification, mentorship, …) | One LLM call per posting against a frozen pre-committed prompt | **Separate agent**, not this plan |
| **L3 — BERTopic + embedding-space** | Discover what's in the data without imposing a list; project the embedding space onto interpretable axes | UMAP → HDBSCAN → c-TF-IDF, plus semantic-axis projection (Bolukbasi/Kozlowski-style), cluster-difference vectors, centroid drift, WEAT-style tests, anchor-neighborhood diffusion | **This plan** |

L1 and L2 answer questions we can pre-register at the level of named families and themes ("did the AI/LLM Engineer family grow?"). L3 answers ones we cannot ("are our families actually distinct density regions in description space?", "what falls outside the closed list?", "where does the embedding space encode AI-vs-traditional, IC-vs-management, builder-vs-operator gradients?", "how have role centroids drifted between 2024 and 2026?").

This plan touches L1/L2 in exactly one place: **§7.7**, a read-only crosstab of BERTopic clusters against L1 modal family and against L2 theme rates, computed once L1/L2 land in `unified_core.parquet`. We report the agreement; we do not tune anything against it.

### 1.2 The four claims this analysis must support

Named here for cross-reference; falsification conditions and predicted signatures are in §1.4.

1. **C1 — Discovery.** AI-native role families have crystallized.
2. **C2 — Decline.** Specific legacy stacks (.NET, COBOL, mainframe, PHP/WordPress, AUTOSAR, ServiceNow, PLC) are shrinking.
3. **C3 — Vocabulary drift.** Existing role clusters are being rewritten in place even when their posting share is stable (DevOps → platform / reliability; QA → CI/CD / integration; cloud → ML / agents).
4. **C4 — Concentration.** Effective number of clusters decreases from 2024 to 2026.

Plus four theoretical-framework claims (T1–T4 in §1.4.2) that L3 numbers feed without being the sole evidence for.

### 1.3 What this analysis is *not* doing

- Not the source of L1 role-family labels — those are produced separately by a closed-list LLM classifier.
- Not the source of L2 skill-theme tags — same.
- Not the source of any AI-rate claim. AI rate uses the canonical `AI_VOCAB_PATTERN` regex on `description_core_llm`. BERTopic and the embedding axes only segment where that AI vocabulary lives and along which dimensions it has moved.
- Not generating direct claims about juniors vs seniors — those go through L1/L2 conditioned on `seniority_final`. BERTopic supplies a topics-per-class view and the embedding-space IC↔management axis as triangulations, not primary instruments.
- Not building or validating the L1/L2 taxonomies. The single touchpoint is §7.7's read-only crosstab.

### 1.4 What this analysis can prove or disprove

The point of laying this out before any code runs is to commit to falsification conditions. Each row below names a load-bearing claim, the L3 signature we'd expect if it's true, and the signature that would force us to walk it back. Both columns are paper-visible: we report whichever way each row lands.

#### 1.4.1 Primary claims (paper §"SWE is changing")

| Claim | Predicted L3 signature | Falsification signature |
|---|---|---|
| **C1 — AI-native role families have crystallized.** | ≥ 1 BERTopic cluster whose c-TF-IDF top terms are dominated by AI/LLM/agent vocabulary, n_2024 small (< 5% of period total), n_2026 large (> 15%); the AI-native↔traditional axis (§6.1) shows period-mean shift ≥ 0.05 cosine units and ≥ 3× anchor leave-one-out sensitivity; anchor-neighborhood diffusion (§6.5) for the AI-engineer anchor grows monotonically across cosine thresholds. | No cluster's top vocabulary is AI-flavored; or n_2026/n_2024 < 2× for the AI-flavored cluster(s); or the AI-axis shift is within anchor sensitivity; or the AI-axis exists in 2024 at comparable mass and has only relabeled itself. |
| **C2 — Specific legacy stacks are shrinking.** | ≥ 1 cluster whose vocabulary is dominated by `.NET`, `COBOL`, `mainframe`, `PHP`/`WordPress`, `AUTOSAR`, `ServiceNow`, or `PLC`, with n_2026/n_2024 < 0.6; per-cluster centroid drift (§6.3) for these clusters is large; WEAT (§6.4) shows legacy clusters significantly less associated with `innovation` attribute set than AI clusters. | Posting share for these stacks is flat or grows; or the vocabulary doesn't form discrete clusters at all (legacy work has dispersed into other clusters rather than shrinking as a class). |
| **C3 — Existing roles are being rewritten in place.** | ≥ 4 stable clusters (n_2024 ≈ n_2026 within 30%) whose c-TF-IDF top-20 changes by ≥ 30% set-difference between periods, AND whose centroid drift vector (§6.3) loads heavily on the AI-native↔traditional axis (§6.1); boundary-fraction (§6.2) between AI Engineer and adjacent clusters grows by ≥ 5pp. | Stable clusters have stable vocabulary and stable centroids — change is happening only at the cluster level, not within-cluster level. The "rewriting in place" framing dies; we tell only the cluster-emergence story. |
| **C4 — Roles are concentrating.** | Effective number of clusters (entropy of share distribution at the headline K) lower in 2026 than 2024 by ≥ 10%, **OR** top-5 cluster share (HHI) higher by ≥ 5pp. Two metrics should agree in direction. | Cluster fragmentation increases or stays flat; the "consolidation" framing in the paper outline doesn't survive and we drop it. |

If C1 fires while C2, C3, or C4 do not, the paper has a one-claim narrative ("AI-engineering boom") not a four-claim one. We commit to reporting whichever subset survives, not pretending the whole frame holds.

#### 1.4.2 Theoretical-framework claims (paper §"State of transition")

| Claim | Predicted L3 signature | Falsification signature |
|---|---|---|
| **T1 — Brynjolfsson J-curve / intangible-investment phase.** | AI-flavored clusters' 2026 vocabulary is future-tense and exploratory ("will use," "exploring," "building toward," "investigating"); high posting volume but low specificity per posting. The IC↔management axis (§6.1) shows AI postings tilting toward abstract framings. | AI vocabulary is concrete and present-tense; AI is integrated alongside other technologies as a routine tool, not as a forward bet. |
| **T2 — Seniority-biased technological change reverses (Hosseini & Lichtinger).** | Junior-leaning clusters shrink less than senior-leaning clusters in routine-coding cells; AI/LLM clusters skew senior in `topics_per_class(seniority_final)`. | Seniority distribution flat across clusters, or junior clusters shrink fastest — the SBTC-reversal framing fails. |
| **T3 — Senior bottleneck persists (Autor & Thompson; Gans & Goldfarb "O-Ring").** | New AI clusters have higher senior share than the corpus average; orchestration / mentorship vocabulary concentrates in senior YoE bins inside those clusters; WEAT (§6.4) shows seniority↔orchestration association strengthens 2024→2026. | New clusters are junior-balanced or junior-tilted — the "AI lifts the codified rungs but seniors persist" thesis loses its support. |
| **T4 — Normal-technology diffusion (Narayanan & Kapoor).** | 2024→2026 cluster restructuring is gradual: many partial movements rather than a few discontinuous jumps; the AI-axis projection distribution shifts continuously rather than bimodally. | A small number of clusters absorb almost all change; the rest are static — a "discontinuous shock" reading rather than a diffusion reading. |

#### 1.4.3 Pessimist-vs-skeptic resolution (paper §"The losers of transition")

The paper's framing names a public disagreement (Karpathy/Amodei "juniors are cooked" vs Garman/Rachitsky "junior demand fine"). L3-testable angles:

- **Junior cluster mix:** topics-per-class on `seniority_final` shows whether the junior posting **mix** in 2026 differs from 2024 even when junior **share** is flat. If the same proportion of postings is junior but those juniors live in different clusters, that's "scope changes, not headcount." The IC↔management axis (§6.1) cross-checks: if junior postings move toward the IC pole and senior toward management/abstract, the paper's resolution writes itself.
- **AI vocabulary diffusion vs concentration:** anchor-neighborhood diffusion (§6.5) for the AI-engineer anchor at multiple cosine thresholds. If the threshold-0.5 neighborhood grows uniformly across all clusters, that's "AI talk is cheap." If neighborhood growth concentrates within dedicated AI clusters, that's "AI work is real."

#### 1.4.4 The AI-cohort sub-structure question (continuation of v3 prior)

Open from `composite_B_v3_findings.md`: is "AI Engineer" splitting into RAG / agents / evals / foundation-model research, or is it still one undifferentiated job? The v3 prior found it had not yet differentiated geometrically (silhouette 0.27, bimodal seed-ARI). This run tests whether four more weeks of 2026 data plus OpenAI 3072-d embeddings change the answer. With L1/L2 produced separately at the role-family level, this question lives entirely on the L3 layer:

- **Differentiated:** hierarchical clustering inside the AI cluster(s) produces ≥ 3 sub-clusters, cross-scale silhouette ≥ 0.4, top-words and LLM-proposed labels distinguish the sub-clusters cleanly; centroid drift (§6.3) within the AI region differs across sub-clusters.
- **Not yet differentiated (v3-consistent):** one mega-cluster, possibly with vertical (healthcare, fintech) splits the geometry enforces but no horizontal (RAG/agents/evals) splits it can find.

A null result here is itself a finding for the paper: it says "the field is naming a specialty before the work has split into sub-specialties."

#### 1.4.5 What this analysis cannot prove

Stated up front so reviewers cannot accuse us of conflating registers. L3 on job descriptions speaks to:

- The **filter** (what employers write down)
- The **vocabulary** (what they call it)
- The **distribution** (how many of each)
- The **geometry** (where the description sits in semantic space)

It does **not** speak to:

- Realized hiring (postings ≠ hires; ghost postings exist)
- Wages or salary distributions
- Actual on-the-job content (descriptions ≠ work; cf. METR slowdown vs Peng et al. speedup, both out of scope here)
- Causation between AI capability and posting change (correlation only)
- Counterfactuals ("what would have happened without AI") — the analysis is descriptive
- Cross-platform validity (LinkedIn-only)

Every causal claim in the paper must reroute through the JD-substrate framing the broader methodology section establishes; L3 results are a richer description, not a different epistemology.

---

## 2. Inputs

### 2.1 Source

`data/unified_core.parquet` — already preprocessed, LLM-classified, embedded.

| Field | Type | Use |
|---|---|---|
| `uid` | str | Row identifier; cluster labels and axis projections keyed on this |
| `source` | str | `kaggle_arshkon` / `kaggle_asaniczka` / `scraped` |
| `period` | str | `2024-01`, `2024-04`, `2026-03`, `2026-04` |
| `title` | str | Used in cluster characterization, *not* directly in embedding (already in `description_core_llm` via Stage 11 input concat) |
| `description_core_llm` | str | LLM-stripped boilerplate-removed body — the c-TF-IDF substrate |
| `job_description_embedding` | float32[3072] | OpenAI `text-embedding-3-large`, computed at Stage 11 from `title + description_core_llm` |
| `is_swe`, `is_control` | bool | Sample selection |
| `seniority_final`, `yoe_min_years_llm` | various | Post-hoc characterization, never input to clustering |
| `company_name_canonical`, `is_aggregator`, `metro_area` | str/bool | Per-firm cap; characterization |
| `date_flag`, `has_llm_extraction`, `has_llm_classification` | various | Filter to clean rows |
| `role_family_l1` | str | **Read-only**, populated by separate L1 classifier; used in §7.7 crosstab only. Not present until that pipeline runs. |
| `skill_theme_*` | bool[8] | **Read-only**, populated by separate L2 classifier; used in §7.7 crosstab only. |

### 2.2 Embedding choice — why OpenAI 3072-d, not MiniLM 384-d

The prior BERTopic run (`eda_archive/scripts/S27_v2_bertopic.py`, archived) used `all-MiniLM-L6-v2` 384-d. It produced a usable taxonomy (29 families) but had two known failure modes documented in `composite_B_v3_findings.md`:

- **Mega-cluster blending.** A single density region housed both AI/ML and traditional data-science postings; embeddings could not separate them, requiring a post-hoc AI-vocab regex split (`0_AI` / `0_nonAI`).
- **Stability ARI 0.44–0.49 across seeds** at family granularity — seed-sensitive enough that one of three seeds produced a materially different layout.

`text-embedding-3-large` at 3072-d is the standard upgrade and is already computed. Two reasons it should help: (a) larger capacity for fine semantic distinctions in technical vocabulary; (b) trained on substantially more code/tech text than MiniLM. The hypothesis it should test is whether the AI/data mega-cluster splits without regex assistance. We should not assume it does — we report whichever way it lands.

We **also re-run the entire pipeline with MiniLM** as one of the embedding ablations (§8) so the paper can claim the result is not embedding-model-specific.

### 2.3 Text substrate

`description_core_llm` (with COALESCE fallback to raw `description` for the ~1% unlabeled rows) for both embedding and c-TF-IDF. This matches the project's substrate convention (`methodology_protocol.md` §1) and means c-TF-IDF top-words are not contaminated by EEO / benefits / recruiter boilerplate that would otherwise dominate cross-period comparisons.

### 2.4 Filters before sampling

- `description_core_llm IS NOT NULL`
- `LENGTH(description_core_llm) >= 200` (drops postings too short to embed reliably)
- `date_flag = 'ok'`
- `has_llm_classification = TRUE` (so we have L1/L2 labels for every clustered posting once L1/L2 land)
- `job_description_embedding IS NOT NULL` (~99.2% retention)

### 2.5 Anchor postings

The semantic-axis analyses (§6.1) and the anchor-neighborhood diffusion (§6.5) require **anchor postings** — short hand-written exemplars representing the poles of each axis or the canonical role descriptions. Anchors are committed in `figures/bertopic/config.py` as Python strings, embedded with the **same** `text-embedding-3-large` model used for postings, and cached alongside posting embeddings in `data/bertopic/embeddings_cache.npy` so anchors and postings share one space and one I/O path. Anchors are **frozen** before any axis is fit; sensitivity to anchor choice is reported via leave-one-out (§6.6).

---

## 3. Sample

### 3.1 Two samples — A (SWE) and B (SWE + Control)

The plan no longer uses an exploration/confirmation split. The split was justified by non-circular L1 validation, and L1 is now produced by a separate agent against a closed list — there is no taxonomy work in this plan to validate. Dropping the split lets BERTopic fit on the full SWE pool, which gives more density, more stable cluster boundaries, and reinstates the within-2024 cross-source placebo (§7.6) as a clean source-confound check.

| Sample | Rows | Used for |
|---|---|---|
| **A — SWE sample** | SWE postings from `kaggle_asaniczka` (2024-01) + `kaggle_arshkon` (2024-04) + `scraped` (2026-03, 2026-04) | Headline BERTopic fit; all paper claims about SWE role-landscape change. |
| **B — SWE + Control** | Sample A plus control occupations from arshkon + scraped | Cross-occupation test. Fit two ways and compared per §3.3. |

#### Sampling cap — per (company × period × normalized title)

The cap prevents prolific firms from dominating cluster centroids. The right unit to cap on is **`company_name_canonical × period × title_normalized`**, not `company × period` alone. Reason: a firm posting 200 SWE roles split across 30 distinct titles is showing real role-mix breadth that we want to preserve, while a firm posting "Senior Software Engineer" 50 times in a week is showing literal duplication that we want to suppress. A pure company cap conflates these.

**Cap value: 5 per `(company_name_canonical, period, title_normalized)`.** Lower than the legacy `30 per (company × period)` because the bucketing is finer — most buckets have far fewer than 5 members, so the cap binds only on actually-duplicated postings.

**Title normalization.** Lowercase, strip non-alphanumeric, collapse whitespace. Groups `Senior Software Engineer` / `senior software engineer` → same bucket; keeps `Senior SWE III` and `Senior SWE II` as distinct (intentional — those are different roles). More aggressive normalization is its own can of worms and not justified for a deduplication cap.

**Within-bucket selection.** `ROW_NUMBER() OVER (PARTITION BY canonical_co, period, title_normalized ORDER BY HASH(uid))`, take rn ≤ 5. Hash-on-uid order randomizes within bucket so we don't preferentially keep postings clustered in time (which would correlate with content for batch-posted reqs).

Aggregator inclusion is the default; aggregator exclusion is a robustness ablation (§8).

#### Sample sizes — confirmed in Stage 0

Pre-cap counts from `unified_core.parquet`: 19,047 (asaniczka SWE) + 5,433 (arshkon SWE) + 35,639 (scraped SWE) = **60,119 SWE rows pre-cap**. Post-cap is the first measured output of Stage 0, written to `intermediate/sample_sizes.csv`. Order-of-magnitude bound: Sample A is ~50–55k post-cap; the embedding tensor (3072-d float32) is ~0.6 GB. Comfortable in any case.

### 3.2 Stratification record

Cluster outputs are reported with the following row-level breakdown alongside every cluster:

- n total, n by period, growth ratio (2026 / 2024)
- AI-vocab rate (broad regex, strict regex)
- Seniority distribution (junior / mid / senior shares; LLM abstention rate)
- Median YOE (where available)
- Top 5 firms; top 3 metros
- Aggregator share
- Within-cluster substrate-length distribution (sanity check vs boilerplate residue)
- L1 modal family + L1 family share table (when L1 has been applied)
- L2 theme rates (when L2 has been applied)

This is the ground truth row that goes into the appendix datasheet — never less detailed than this.

### 3.3 Sample B — fit two ways, pick by objective criteria

There are two legitimate ways to bring control into the analysis:

- **B-joint.** Fit a fresh BERTopic on Sample B (SWE + Control). Produces one taxonomy that includes occupation-level structure (nurse, civil engineer, accountant clusters).
- **B-projected.** Fit BERTopic on Sample A only; project Sample B's control postings onto the existing model via `topic_model.transform(control_docs, control_embeddings)`. Produces topics-per-class on the SWE-derived structure.

B-joint may over-fragment SWE structure (the model spends capacity on cross-occupation distinctions); B-projected may force control postings into ill-fitting SWE clusters. Run both. Pick the headline variant by:

- **SWE-cluster preservation.** ARI between Sample A's BERTopic clusters and the SWE rows under each B variant — measures how much SWE structure each variant retains.
- **Label distinctness.** Number of distinct LLM-proposed labels at the headline K after deduplicating on label-embedding cosine ≥ 0.85 — flags a B variant where many clusters end up redundant.
- **Topic coherence.** Mean C_v across clusters (§7.8) — flags incoherent splits.

The variant that wins on at least two of three is the headline; the loser appears in the appendix as a sensitivity check. Ties are deferred to the labeling protocol (TBD, see §5).

---

## 4. Method — BERTopic

### 4.1 Pipeline

Standard BERTopic, with our pre-computed embeddings:

```
job_description_embedding (3072-d, OpenAI text-embedding-3-large)
   └─ UMAP (3072 → 5)              # cosine metric, seeded
       └─ HDBSCAN (min_cluster_size, min_samples)
           └─ c-TF-IDF on description_core_llm   # cluster representation
               └─ KeyBERTInspired              # representation refinement
                   └─ MaximalMarginalRelevance # n-gram diversification (diversity=0.3)
                       └─ OpenAI gpt-5.5 / gpt-5.4-mini  # one-line cluster label (§5)
```

UMAP reduction is needed because HDBSCAN's distance computation is poorly behaved in 3072-d (curse-of-dimensionality, distance concentration). 5-d is the BERTopic default and matches our prior run; 10-d is an ablation.

### 4.2 Hyperparameters

Pre-registered. Frozen before fitting; ablations (§8) test sensitivity.

| Component | Param | Value | Why |
|---|---|---|---|
| UMAP | `n_neighbors` | 15 | BERTopic default; preserves local + some global structure |
| UMAP | `n_components` | 5 | BERTopic default; HDBSCAN-friendly |
| UMAP | `min_dist` | 0.0 | Encourage tight clusters for HDBSCAN |
| UMAP | `metric` | cosine | OpenAI embeddings are L2-normalized; cosine is the appropriate metric |
| UMAP | `random_state` | 42 (primary), 1337, 2026 (stability) | 3-seed protocol matches v3 prior |
| HDBSCAN | `min_cluster_size` | 30 (initial), tuned via §4.6 | At ~50k SWE rows in Sample A, 30 targets ~80–120 raw topics before reduction; finalized empirically |
| HDBSCAN | `min_samples` | round(0.5 × min_cluster_size) | Pinned rule, not swept; standard |
| HDBSCAN | `metric` | euclidean | Acts on UMAP output, not raw embeddings |
| HDBSCAN | `cluster_selection_method` | `eom` | Excess-of-Mass; matches prior |
| HDBSCAN | `prediction_data` | True | Needed for outlier reduction and approximate_distribution |
| CountVectorizer | `ngram_range` | (1, 3) | Single tokens + bigrams + trigrams capture multi-word skills ("vector database") |
| CountVectorizer | `min_df` | 10 | Drop hapaxes that c-TF-IDF will overweight |
| CountVectorizer | `max_df` | 0.4 | Drop terms in >40% of clusters (corpus-generic) |
| CountVectorizer | `stop_words` | English + custom | Custom list drops residual boilerplate tokens (see §4.5) |
| CountVectorizer | `token_pattern` | `(?u)\b[a-zA-Z][a-zA-Z\-\+/\.]+\b` | Keeps `c++`, `node.js`, `ci/cd`, `.net` |
| MMR representation | `diversity` | 0.3 | Slight bias toward diverse top-words across clusters; prevents top-10 lists overlapping heavily |
| BERTopic | `min_topic_size` | match HDBSCAN `min_cluster_size` | Single source of truth |
| BERTopic | `calculate_probabilities` | False | Use `approximate_distribution` post-hoc instead — much cheaper |
| BERTopic | `nr_topics` | None initially; sweep K post-fit per §4.4 | K is characterized, not pre-fixed |

The `min_cluster_size` initial value (30) sits in the middle of the §4.6 sweep grid; the headline value is finalized by the criteria stated there, not by the initial guess.

### 4.3 BERTopic features we use

| Feature | Purpose | Paper artifact |
|---|---|---|
| **Custom embeddings** (`embeddings=` kwarg in `fit_transform`) | Pass our precomputed OpenAI vectors directly; no in-pipeline encoding | All clustering |
| **Hierarchical topics** (`hierarchical_topics(docs)`) | Build dendrogram of cluster mergers; pick a level for "super-family" view | F2 (super-family vs family resolution); §1.4.4 AI-cohort hierarchy |
| **Topics over time** (`topics_over_time(docs, timestamps)`) | Per-period topic shares without re-fitting | F3 drift; T2 growth/decline |
| **Topics per class** (`topics_per_class(docs, classes=...)`) | Topic shares conditioned on class (SWE-vs-control; junior-vs-senior; aggregator-vs-direct) without refitting | F6 junior/senior topic split; §7 robustness |
| **Topic reduction** (`reduce_topics(docs, nr_topics=K)`) | Merge similar topics to a target K | Sweep K per §4.4 |
| **Outlier reduction** (`reduce_outliers(..., strategy="embeddings")`) | Reassign HDBSCAN noise points to nearest topic | Sensitivity: shares **with and without** outlier reassignment |
| **Approximate distribution** (`approximate_distribution(docs)`) | Soft topic membership per doc; needed for entropy claims | C4 concentration |
| **`find_topics(query, top_n=5)`** | Semantic search: pass a seed phrase, get top topics | Sanity-check that L1 families exist as data-driven topics |
| **OpenAI representation model** | LLM generates a one-line topic label | §5 cluster naming |
| **`get_document_info(docs)`** | Per-doc: topic, prob, representative status | Per-row artifact `assignments.parquet` |

Features we deliberately **do not** use:

- `online_topic_model` / `online_dim_reduction` — corpus is static.
- `class_model.fit` (supervised topic modeling) — would defeat discovery purpose.
- Built-in BERTopic visualizations — Plotly HTML; the paper requires `figures/style.py`.
- `merge_models` / `merge_topics` based on multiple fits — risk of post-hoc taxonomy fishing.

**Honesty about "topic" terminology.** BERTopic does not perform probabilistic topic modeling in the LDA sense — it is a clustering wrapper over UMAP+HDBSCAN with c-TF-IDF for cluster labeling. Calling its outputs "topics" is convenient but blurs methodology. The paper's methods subsection writes "data-driven cluster" or "role family cluster" rather than "topic" wherever the distinction matters.

**Outlier-reduction strategy choice.** BERTopic offers four strategies (`c-tf-idf`, `embeddings`, `distributions`, `tokenset_similarity`). We use `embeddings` as the headline strategy — reassign each noise point to the topic whose centroid (in UMAP space) is nearest in cosine. Reasons: (a) consistent with the rest of the pipeline, which is geometry-driven; (b) `c-tf-idf` over-weights short postings; (c) `distributions` requires `calculate_probabilities=True`. We report cluster shares both with and without outlier reduction (§7.9).

**Approximate-distribution parameters.** When computing soft topic distributions per document via `approximate_distribution(docs)`, pin: `window=8`, `stride=4`, `min_similarity=0.1`, `padding=False`. Pinned in `config.py`.

### 4.4 Topic-count K — sweep, characterize, then commit

K is not a single pre-registered value. Different K answers different questions: small K shows the super-family structure; large K shows fine sub-archetypes. We characterize what changes across K rather than committing in advance to one resolution.

**Sweep grid (pre-registered):** K ∈ {10, 15, 20, 25, 30, 40, 50, 75}. Each K is generated by `reduce_topics(docs, nr_topics=K)` from the same raw fit, so cluster identities are nested.

**For each K, record:** number of clusters surviving; noise rate; mean inter-cluster centroid cosine; mean intra-cluster spread; DBCV score on UMAP output (if computable); seed-pair ARI at this K; per-period reproduction centroid alignment (§7.3); manual interpretability rating from a 5-cluster sample (1–5 scale, author-rated; the labeling protocol per §5 is TBD and supersedes this scratch rating once it lands).

**Headline K — selection criterion (pre-registered).** The smallest K satisfying:

1. Seed-pair ARI ≥ 0.4
2. Per-period reproduction centroid alignment ≥ 0.85
3. Mean interpretability rating ≥ 3.5 / 5
4. Outlier rate ≤ 40%

If multiple K satisfy all four, pick the smallest. If none does, headline K is reported as "no stable family resolution" and the paper makes only super-family-level claims.

**Two K reported in paper, regardless of headline pick:** a **super-family K** (somewhere in {10, 15}) for the broad map figure (F1), and the **headline K** for per-cluster claims (T1).

### 4.5 Custom stopwords

In addition to sklearn's English stopwords, drop:

- Substrate residue: `description`, `responsibilities`, `qualifications`, `requirements`, `position`, `role`, `team`, `work`, `looking`, `seeking`, `candidate`, `experience`
- Boilerplate that survived L9 stripping: `equal`, `opportunity`, `affirmative`, `disability`, `veteran`, `protected`, `sexual`, `gender`, `race`, `religion`, `applicant`, `employee`
- Generic recruiter-CTA: `apply`, `resume`, `interview`, `hiring`, `recruit`, `recruiting`
- Pure-noise tokens we observed in the v3 run: `etc`, `eg`, `ie`, `including`, `e.g`, `i.e`

**Frozen before fitting.** If we discover during analysis that another token dominates a cluster in an uninformative way, we document the finding — we do not silently grow the stopword list.

### 4.6 `min_cluster_size` sweep

Before the K sweep, we tune `min_cluster_size ∈ {10, 20, 30, 50, 70}` on Sample A and record:

- Raw topic count (pre-reduction)
- Noise rate
- Mean inter-cluster cosine distance
- DBCV score (if computable)
- Post-`reduce_topics(30)` ARI vs neighboring sweep points

**Headline `min_cluster_size`** is the value at which (a) post-reduction ARI vs neighboring sweep points is ≥ 0.7 (stable plateau), AND (b) noise rate is in 15–35%. If two values qualify, pick the larger (more conservative). Committed in `config.py` before the K sweep runs.

---

## 5. Cluster naming

A cluster called "AI Engineer" instead of "Data + AI/ML hybrid" changes how every downstream claim reads. This section pins the LLM-proposed labeling step. The full human-review protocol is being defined separately and will be added before headline labels are frozen for the paper.

### 5.1 Three name signals per cluster

For each topic at the headline K (and additionally for clusters at the super-family K):

1. **c-TF-IDF top-15 terms.** Pure data-driven.
2. **KeyBERTInspired top-10 phrases.** Filters c-TF-IDF terms by similarity to the topic centroid embedding.
3. **OpenAI LLM label.** Prompt: "You are labeling a cluster of software-engineering job descriptions. Given these top words: {c-TF-IDF top-15} and these representative posting excerpts (titles + first 200 chars of description_core_llm): {5 exemplars}, produce a 2–4 word noun-phrase label that names the role family or sub-archetype. Do not invent vocabulary not present in the words or excerpts. Output JSON: {label, confidence: 0–1, alternative}."

**Model selection.** Use **`gpt-5.5`** (snapshot pinned at `gpt-5.5-2026-04-23`) when the total number of LLM calls in a single pass is < 100 (cluster naming at headline K + super-family K is ~30–50 calls — comfortably under). Switch to **`gpt-5.4-mini`** for any pass that exceeds 100 calls. Both model IDs are pinned in `figures/bertopic/config.py`; switching mid-pass is forbidden. Authentication uses the same `~/.config/job-research/openai.env` mechanism as the preprocessing pipeline.

**Exemplars** are selected as the 5 documents with highest `topic_model.get_representative_docs(topic_id)`, supplemented with 2 random members so the prompt sees both prototypes and breadth.

### 5.2 Final labeling protocol — TBD

The protocol that turns the LLM proposal + top-words + exemplars into the frozen paper labels is being defined separately and will be added to this section before headline labels are committed. Until then, **LLM-proposed labels are working labels** — they appear in intermediate artifacts and exploratory figures but are not the artifact released in the paper. No claim in the paper depends on a label that has not been through the final protocol.

### 5.3 Frozen-name commit

Once §5.2 protocol completes, **labels are frozen**. They appear verbatim in every figure, table, and prose mention. Any later change to a label requires a paper-visible "Erratum: cluster X relabeled from Y to Z because …" note. This avoids the failure mode where labels drift to support claims.

---

## 6. Embedding-space analyses

This section specifies five complementary analyses that operate on the 3072-d embedding space directly, going beyond what BERTopic clustering alone reveals. The methodology adapts Bolukbasi et al. (2016)'s direction-identification primitive, Kozlowski et al. (2019)'s cultural-axis projection, Caliskan et al. (2017)'s WEAT statistical test, Garg et al. (2018)'s temporal-drift framing, and Grand et al. (2022)'s formalization of semantic projection. All five are exploratory — the §13 critical-evaluation gate decides which become paper artifacts and which are dropped.

**Caveat that frames the section.** Bolukbasi-style analogy arithmetic (`king − man + woman ≈ queen`) was developed for word2vec/GloVe and composes reasonably there because of the (log-)bilinear training objective. Our embeddings are sentence-level (`text-embedding-3-large`, 3072-d, full posting), where vector arithmetic is empirically less reliable. **We lean on projection (robust); we do not claim sentence-level analogies (brittle).** Reviewers familiar with the literature will know the difference; the paper's prose acknowledges it explicitly.

### 6.1 Semantic axis projection

Define K axes from anchor postings, project every posting onto each axis, characterize the period-mean shift and per-cluster axis profile.

**Axis construction.** For each axis, hand-write 6 anchor postings per pole, embed them with `text-embedding-3-large`, form the difference set `D = {v(positive_i) − v(negative_i) : i ∈ 1..6}`, take the first principal component of `D`. This is the axis vector `g`. Bolukbasi's PCA-over-differences move, transferred to sentence-level. Using PCA rather than a simple mean buys robustness to one or two noisy anchor pairs.

**Projection.** For any posting `w`, the axis score is `proj(w) = cos(w, g)`. Range [-1, 1]; positive = aligned with the positive pole, negative = aligned with the negative pole, magnitude = strength.

**Pre-registered axes (5):**

- **AI-native ↔ traditional**
- **IC ↔ management**
- **Builder ↔ operator**
- **Concrete ↔ abstract**
- **Generalist ↔ specialist**

The exact anchor postings per pole are listed in §11 and committed in `config.py` before axis fitting. They cover the gradients we expect to be informative about C1, C3, T1, T3, and §1.4.3.

**Reported artifacts:**

- Per-axis distribution of posting scores by period (2024 vs 2026) — F9 candidate
- Per-cluster axis profile: cluster mean ± IQR on each axis, sorted by mean — T7
- Axis-mean shift table: 2026 mean − 2024 mean per axis, with permutation null and anchor leave-one-out sensitivity bands

**Decision criterion (Gate 2).** Period-mean shift on the axis ≥ 0.05 cosine units AND ≥ 3× the leave-one-out anchor sensitivity. Falls below threshold → axis cut from paper.

### 6.2 Cluster-difference vectors and boundary postings

For pairs of BERTopic clusters `(A, B)` of substantive interest (e.g., AI Engineer vs Backend; AI Engineer vs Data Scientist), compute the centroid difference `δ_AB = v_A − v_B`. Project every posting onto `δ_AB`. Postings with low absolute projection are **boundary postings** — semantically ambiguous between A and B.

**Reported artifacts:**

- Boundary fraction per cluster pair, per period: postings with |proj| < 0.05 (in cosine units, since `δ_AB` is normalized)
- Δ boundary fraction 2024→2026 per pair — geometric "blurring" measure

**Decision criterion (Gate 2).** Boundary-fraction change ≥ 5pp AND permutation p < 0.05 against the null of period-shuffled posting labels. Falls below → cut.

**Why this matters for the paper.** Maps onto C3 (within-cluster drift) and onto §1.4.3 (pessimist-vs-skeptic resolution: "scope changes, not headcount"). If role boundaries are dissolving, the boundary fraction is the cleanest geometric signal we have.

### 6.3 Centroid drift over time

For each BERTopic cluster, compute `Δ = v_cluster_2026 − v_cluster_2024`. The direction `Δ` encodes what semantically changed about that work between the two periods. This is the Garg et al. (2018) move adapted from word-level (occupation embeddings over 100 years) to cluster-level (cluster centroids over two years).

**Decomposition onto pre-registered axes.** Project `Δ` onto each §6.1 axis to identify *which axis* the drift loaded onto. This makes the per-cluster narrative concrete: "the Backend cluster's 2024→2026 drift loads heavily on the AI-native↔traditional axis (cos 0.42), modestly on the builder↔operator axis (cos −0.11), negligibly elsewhere."

**Control-occupation differencing.** Compute the analogous `Δ_control` for control occupations in Sample B. The SWE-specific component of drift is `Δ_swe − Δ_control` (in vector terms; project both onto the same axes for comparability). This guards against "everything drifted because the corpus drifted" confounds.

**Reported artifacts:**

- Per-cluster drift magnitude `|Δ|` — F4 supplementary panel
- Per-cluster drift decomposition table: axis loadings — T8 candidate

**Decision criterion (Gate 2).** Drift magnitude ≥ 2× the control-occupation drift on the same axis. Drift dominated by control-shared signal → cut from paper.

### 6.4 WEAT-style association tests

Pre-register 3–5 attribute-pair tests of the Caliskan et al. (2017) form. Each test compares two target sets X, Y against two attribute sets A, B by their differential cosine similarity, with effect size (Cohen's d) and permutation p-value.

**Pre-registered tests:**

1. **AI vs legacy clusters × innovation vs maintenance attributes.** Are AI-Engineer cluster centroids more associated with innovation-vocabulary anchor sentences than legacy-stack cluster centroids are?
2. **2026 vs 2024 SWE postings × orchestration+mentorship vs solo-IC attributes.** Has the SWE corpus shifted toward orchestration framings between the two periods?
3. **Senior vs junior SWE postings × architecture+design vs implementation attributes.** Is the senior-architecture association stronger in 2026 than in 2024?
4. **AI clusters × growth vs stability attributes.** Are AI clusters more growth-vocabulary aligned than the corpus average?
5. **AI clusters × exploration vs exploitation attributes.** T1 J-curve test: are AI clusters more exploratory in vocabulary than other clusters?

**Statistical procedure.** For each test, compute Cohen's d of the differential cosine; compute p via 10,000-sample permutation null. Bonferroni-correct across the five tests.

**Decision criterion (Gate 2).** Cohen's d ≥ 0.5 AND Bonferroni-corrected p < 0.01. Tests not clearing the bar → reported as null in the paper, not cut entirely (a null is informative when pre-registered).

### 6.5 Anchor-neighborhood diffusion

Define ~5 anchor descriptions, embed each, compute its k-nearest neighbors in 2024 and in 2026 separately at multiple cosine thresholds. The neighborhood size and composition over time tells the diffusion-vs-concentration story (§1.4.3).

**Pre-registered anchors:**

- Canonical AI Engineer ("a SWE who builds RAG agents and integrates LLMs into production")
- Canonical Frontend Engineer ("a SWE who builds React UIs and ships features to web users")
- Canonical Backend Engineer ("a SWE who designs APIs and operates server infrastructure")
- Canonical Legacy Specialist ("a SWE who maintains a COBOL mainframe codebase")
- Canonical SRE ("a SWE who runs reliability for production systems")

**Cosine thresholds (pre-registered):** {0.5, 0.6, 0.7, 0.8}. Reporting trend across all four.

**Reported artifacts:**

- Per-anchor neighborhood size by period at each threshold — F10 candidate
- Per-anchor neighborhood composition by BERTopic cluster — supplementary

**Decision criterion (Gate 2).** Trend monotonic in the same direction across all four cosine thresholds. Threshold-dependent results → cut.

### 6.6 Pre-registration discipline for embedding-space analyses

The big risk in this section is anchor-set fishing — choosing anchors that produce the axis we wanted. Mitigations to commit to before any axis is fit:

- **Anchors committed in `config.py`** with rationale per anchor. PR-tracked. No silent changes; any change requires a paper-visible erratum entry in `prereg_log.md`.
- **Anchor leave-one-out sensitivity** for every reported axis: report the mean axis vector and the cosine spread when each anchor (one at a time) is dropped. If an axis's identity shifts > 0.1 cosine on any single-anchor leave-out, the axis is unstable and gets a paper-visible flag.
- **Held-out anchor validation:** hold out 30% of each anchor set, project them onto the axis fit on the 70%. Held-out positives should land on the positive pole and held-out negatives on the negative pole. Report the hit rate.
- **Permutation null:** project on 1,000 random unit directions; the axis effect size should sit in the > 95th percentile.
- **Anchor-set release:** every anchor used in any reported axis is published in the paper appendix and in the release tarball. Reviewers can read every word that shaped every axis.

The §13 sub-agent for §6.1 runs all four mitigations as part of its standard task spec, not optionally.

---

## 7. Validation and rigor

For an AIES paper, we need to defend that the topic structure is real, not an artifact of seeds, embeddings, or sample. The defense is layered.

### 7.1 Stability — three seeds (Stage 1 gate)

Three seeds: 42, 1337, 2026. Same data, same hyperparameters, different UMAP `random_state`. **This is a Stage 1 gate (§13.2)**: if the headline K's seed-pair ARI < 0.4 AND centroid alignment < 0.85, Stage 2 sub-agents do not fan out — we fall back to super-family granularity per §10.5 instead of building on an unstable foundation.

Report:

- **Topic-level ARI** between seed pairs (raw fits, before `reduce_topics`). Baseline: prior MiniLM run scored 0.49.
- **Family-level ARI** at headline K (and at K=30 specifically, for direct comparability with the v3 prior). Baseline: prior 0.44 at K=30.
- **Centroid alignment.** For each seed pair, find the best 1-to-1 cluster matching by Hungarian algorithm on centroid cosine, then report the mean centroid cosine of matched pairs. This catches the case where ARI is low because cluster boundaries shifted but centroids are stable.

If any pair has ARI < 0.4 AND centroid alignment < 0.85, we say so plainly in the paper. We do not pick the friendliest seed.

### 7.2 Stability — bootstrap resampling

Three bootstrap samples of Sample A at 80% (without replacement, stratified by period and source). For each, refit BERTopic, then reduce to the headline K. Report ARI vs the headline fit on the overlapping rows. This addresses the "is the taxonomy a function of the specific 50k rows" worry.

### 7.3 Per-period reproduction

The prior v3 run used this and it was the load-bearing stability check. Refit BERTopic on **2024-only** (asaniczka + arshkon) and **2026-only** (scraped) subsets of Sample A (independent fits), reduce each to the headline K. For every joint-Sample-A cluster, find the nearest period-fit cluster by centroid cosine; report mean and median.

If joint-fit clusters reproduce on time-sliced data with mean centroid cosine ≥ 0.85, the cross-period comparison is trustworthy. The v3 baseline was 0.94/0.95.

### 7.4 Cross-method comparison — NMF baseline

Fit NMF (k = headline K) on TF-IDF of `description_core_llm` (same vectorizer settings as c-TF-IDF). Hard-assign by `argmax`. Report ARI / NMI vs BERTopic. Different mathematical machinery → if they roughly agree, the structure is robust to method.

### 7.5 Cross-embedding comparison — MiniLM rerun

Refit BERTopic with `all-MiniLM-L6-v2` 384-d (the v3 prior's embedding) on Sample A. Reduce to the headline K. Compute ARI / NMI vs OpenAI fit. Two outcomes both fine for the paper:

- **Strong agreement (ARI ≥ 0.5).** Cluster structure is not embedding-specific.
- **Weak agreement (ARI < 0.3).** Cluster structure is embedding-specific — we name *which embedding* in every claim and are explicit that `text-embedding-3-large` reveals structure MiniLM does not.

### 7.6 Within-2024 cross-source placebo

The original placebo, reinstated now that asaniczka is no longer reserved for exploration. Fit BERTopic separately on **asaniczka-2024** SWE rows and on **arshkon-2024** SWE rows. Reduce each to the headline K. Compute centroid alignment between the two via Hungarian matching.

If the two within-2024 fits agree (centroid cosine ≥ 0.85), cross-period (2024→2026) effects are not source-confounded. If they disagree, the source-confound caveat is paper-visible.

### 7.7 Convergent validity vs L1 / L2 — read-only, when those labels land

Once `role_family_l1` and `skill_theme_*` columns are populated in `unified_core.parquet` by the separate L1/L2 agent, run a one-shot crosstab:

- **L1 modal-family share per BERTopic cluster.** For each BERTopic cluster, the most-common L1 family and its share. Mean modal share across clusters is the headline number.
- **L2 theme rate per BERTopic cluster.** Per (cluster, theme), the share of postings tagged with that theme. Sortable, reads as a heat panel.

**Read-only discipline:** we do not tune BERTopic or L1/L2 against this crosstab. The crosstab reports the agreement; if the agreement is poor, both sides report what they found and the paper discusses why (different epistemologies, different objectives). One sentence in the paper: "BERTopic clusters and L1 LLM labels show consistent / inconsistent structure (mean modal-family share = X%)."

If the L1/L2 columns are not yet populated by the time Stage 2 runs, this task is queued for a later session and the paper either notes the absence or runs the crosstab at revision time.

### 7.8 Topic quality — coherence and diversity

Standard topic-modeling quality metrics, computed on c-TF-IDF top-10 terms per cluster, using Sample A as the reference corpus:

- **NPMI** (normalized pointwise mutual information). Range [-1, 1]; higher is better.
- **UMass** (log-conditional probability). Range (-∞, 0]; closer to 0 is better.
- **C_v** (sliding-window cosine over normalized PMI). Range [0, 1]; the metric that correlates best with human topic-quality judgments per Röder et al. 2015. Compute via `gensim.models.CoherenceModel(coherence='c_v')`.
- **Topic diversity** (Dieng et al. 2020): fraction of unique tokens across all topics' top-10 lists.

Report mean, median, and distribution per metric. Targets are aspirational, not pre-registered cutoffs:

| Metric | "Reasonable" range |
|---|---|
| NPMI | mean ≥ 0.05 |
| C_v | mean ≥ 0.45 |
| Topic diversity | ≥ 0.6 |

### 7.9 Cluster-size distribution and silhouette

- **Cluster size distribution** at the headline K. Report median, IQR, 5/95 percentiles. Lopsided distributions (single cluster > 30%) flag the v3 mega-cluster failure mode — also handled as a Stage 1 gate (§13.2).
- **Silhouette score** in 5-D UMAP space. Report mean per cluster and overall mean. v3 prior reported 0.27 inside the AI cohort — known weak; we treat ≥ 0.4 as a strong-separation marker.

### 7.10 Honest noise rate

Report the HDBSCAN noise rate before and after `reduce_outliers`. Both numbers go in the paper. Do not silently use `reduce_outliers` to make the noise rate look smaller — that strategy reassigns ambiguous postings to whichever cluster is geometrically nearest, which may not reflect their content.

### 7.11 Cross-model cluster-name robustness

Re-run §5.1 LLM naming with **`gpt-5.4-mini`** as a second labeler against the **`gpt-5.5`** primary (both within OpenAI; same prompt and exemplars). Compute:

- **Exact-match label rate** across clusters
- **Label-embedding cosine** (mean) — labels embedded with `text-embedding-3-large`; cosine ≥ 0.85 counts as semantically equivalent

If exact-match rate < 50% or mean cosine < 0.85, the LLM-proposed labels are model-sensitive and we surface this in the paper alongside the cluster catalog. The check is a sanity floor on the proposal step; the final labels are decided by §5.2 protocol regardless.

---

## 8. Ablations and sensitivity

Two tiers: **primary characterization** (K and `min_cluster_size` — both swept and reported in the body) and **secondary ablations** (single perturbations, appendix only). Each ablation reports ARI vs headline plus a qualitative description of the largest cluster movement.

### 8.1 Primary characterization (in-body)

| Parameter | Grid | Reported as |
|---|---|---|
| **Topic count K** | {10, 15, 20, 25, 30, 40, 50, 75} | Sweep curve: per-K stability ARI, interpretability rating; body figure F7 |
| **HDBSCAN `min_cluster_size`** | {10, 20, 30, 50, 70} | Sweep table per §4.6; body if headline value is contentious, appendix otherwise |

### 8.2 Secondary ablations (appendix Table T4)

| Ablation | Variant | Reason |
|---|---|---|
| **Embedding model** | MiniLM-L6, jobBERT-v2, OpenAI text-embedding-3-small | Embedding-specificity (also §7.5) |
| **UMAP `n_components`** | 5 (headline), 10, 15 | Compression sensitivity |
| **UMAP `n_neighbors`** | 15 (headline), 30, 50 | Local-vs-global structure |
| **Sample cap** | 5/(co × period × title) (headline), 3 and 10 per same bucket, plus the legacy 30/(co × period), plus uncapped | Prolific-employer distortion; title-aware vs title-blind |
| **Aggregator** | Include (headline) vs exclude | Aggregator-mediated content |
| **Substrate** | `description_core_llm` (headline) vs raw `description` | Boilerplate residue |
| **Length floor** | 200 (headline), 100, 400 | Short-posting noise |
| **Outlier reduction** | Off (headline), on (`distributions` strategy) | Soft-vs-hard cluster shares |
| **Sample B variant** | B-projected (headline if it wins §3.3), B-joint (else) | Cross-occupation strategy |
| **Anchor leave-one-out** (axis-based analyses) | Each anchor dropped one at a time | §6.6 anchor sensitivity |

Ablations live in appendix Table T4. We commit to publishing the table whether or not it's friendly.

---

## 9. Deliverables

### 9.1 Artifacts (machine-readable)

| Path | Contents |
|---|---|
| `data/bertopic/model.bertopic` | Fitted BERTopic model on Sample A. Frozen at Stage 1.5 mini-gate. |
| `data/bertopic/assignments.parquet` | (uid, topic_id, topic_label, prob, is_outlier) — primary integration artifact, joined back to `unified_core.parquet` for downstream figures. |
| `data/bertopic/topic_info.parquet` | One row per topic: id, label, n, n_2024, n_2026, growth, c_tf_idf_top, keybert_top, gpt55_label, gpt54mini_label, label_cosine. (`label` is the frozen paper label per §5.3 once the protocol lands; until then, `label` = `gpt55_label`.) |
| `data/bertopic/topics_over_time.parquet` | (topic_id, period, n, c_tf_idf_top_5) — for vocabulary-drift figure |
| `data/bertopic/topics_per_class.parquet` | (topic_id, class_var, class_value, n) — for SWE-vs-control and junior-vs-senior splits |
| `data/bertopic/hierarchy.parquet` | Dendrogram from `hierarchical_topics` — for super-family figure and AI-cohort sub-structure |
| `data/bertopic/k_sweep.parquet` | One row per K in §4.4 grid: K, n_clusters, noise_rate, seed_ari, period_alignment, interp_rating |
| `data/bertopic/stability.parquet` | Per-pair ARI / NMI / centroid-alignment for §7.1, §7.2, §7.3, §7.6 |
| `data/bertopic/ablations.parquet` | One row per §8 ablation: name, ARI vs headline, noise rate, n topics |
| `data/bertopic/embeddings_cache.npy` | Cache of OpenAI embeddings used (postings + anchors), uid- and anchor-id-keyed |
| `data/bertopic/axes.parquet` | One row per axis (§6.1): id, name, anchor_ids, axis_vector_path, leave_one_out_sensitivity, held_out_hit_rate, permutation_null_percentile |
| `data/bertopic/axis_projections.parquet` | (uid, axis_id, projection) — every posting on every axis |
| `data/bertopic/cluster_axis_profile.parquet` | (cluster_id, axis_id, mean, q25, q75) — per-cluster axis profile |
| `data/bertopic/centroid_drift.parquet` | (cluster_id, drift_magnitude, drift_axis_loadings_dict, swe_minus_control) |
| `data/bertopic/boundary_postings.parquet` | (uid, cluster_pair, projection) for cluster pairs of substantive interest |
| `data/bertopic/weat_results.parquet` | One row per WEAT test: name, X, Y, A, B, cohens_d, p_value, p_bonf |
| `data/bertopic/anchor_neighborhoods.parquet` | (anchor_id, period, threshold, neighborhood_size, top_clusters_dict) |
| `data/bertopic/l1_l2_crosstab.parquet` | §7.7 read-only crosstab, present if and only if L1/L2 columns exist |

### 9.2 Figures (paper, all via `figures/style.py`)

- **F1 — UMAP map.** 2×2 small multiples: rows (period: 2024 / 2026) × columns (sample: SWE / SWE+Control). Points colored by BERTopic family at headline K. `FIGSIZE_DOUBLE`.
- **F2 — Hierarchy.** Dendrogram of headline-K clusters merging up to the super-family K.
- **F3 — Growth/decline.** Diverging horizontal bars: per-cluster Δ posting share (2026 − 2024).
- **F4 — Vocabulary drift within stable clusters.** 4 typographic call-outs (no axes): `entered` and `exited` c-TF-IDF terms.
- **F5 — Concentration.** Two panels: Lorenz curve of cluster-share by period; effective number of clusters over the four period bins.
- **F6 — Topics per seniority.** Heatmap: cluster × {junior, mid, senior}.
- **F7 — K sweep.** Body figure characterizing how stability ARI and noise rate move across K.
- **F8 — AI-cohort hierarchical zoom.** §1.4.4 — sub-cluster silhouette, per-sub-cluster c-TF-IDF, `topics_over_time` *inside* the AI region across the four period bins.
- **F9 — Axis projection panel.** §6.1 — five small multiples, one per axis: distribution of posting scores by period (2024 vs 2026 overlay), with cluster centroids annotated as small markers.
- **F10 — Anchor-neighborhood diffusion.** §6.5 — line chart, neighborhood size by period at four cosine thresholds, one panel per anchor.

F8–F10 are conditional on the §13 critical-evaluation gate. Figures that fail the gate do not enter the paper.

### 9.3 Tables

- **T1 — Cluster catalog.** Rows: headline-K clusters. Cols: label, top-words, n_2024, n_2026, growth, AI-vocab rate, median YOE, top firms, top metros, L1 modal family (when L1 lands).
- **T2 — Stability.** Seed pairs, bootstrap pairs, per-period reproduction, within-2024 cross-source — all centroid-alignment and ARI numbers in one table.
- **T3 — Convergent validity.** ARI / NMI: BERTopic-vs-NMF, OpenAI-vs-MiniLM, joint-vs-2024-only, joint-vs-2026-only, asaniczka-2024-vs-arshkon-2024, plus L1/L2 crosstab (when those land).
- **T4 — Ablations (full).** §8.2 secondary ablations, exhaustive. Appendix.
- **T5 — Naming proposal record.** Per cluster: c-TF-IDF top-3, KeyBERTInspired top-3, `gpt-5.5` proposed label, `gpt-5.4-mini` proposed label, label-embedding cosine. Frozen paper label (added once §5.2 lands) is appended as a final column.
- **T6 — Robustness landscape.** §9.5 sign-consistency matrix: rows = headline + each §8.2 ablation, cols = primary claims C1–C4, cells = ✓ / ✗ / partial / N/A. **In-body**, the only ablation summary that reaches the body of the paper.
- **T7 — Cluster axis profile.** §6.1 — rows: clusters; cols: 5 axes; cells: cluster mean axis score (with IQR shaded). Compact way to read what each cluster *is* on the embedding gradients.
- **T8 — Centroid drift decomposition.** §6.3 — rows: clusters with substantial drift; cols: 5 axes; cells: drift loading per axis. Tells the C3 story per cluster.
- **T9 — WEAT results.** §6.4 — one row per pre-registered test: targets, attributes, Cohen's d, p_bonf.

T7–T9 are conditional on the §13 critical-evaluation gate.

### 9.4 Prose deliverables

- A 1–2 page methods subsection in the paper, drawn from §1–§7 of this doc. Same caveats, half the words. Must call out the L3-as-discovery framing and the L1/L2-handled-elsewhere framing.
- Two appendix sections: pre-registration log (§11), full ablation table (§8).

### 9.5 Presenting ablations in the paper

For an AIES paper, ablations should be **doubly visible**: a small in-body summary that reviewers can read in 30 seconds, plus the full table in the appendix. Three artifacts working together:

**1. The ablation table (T4) — exhaustive, appendix.** One row per perturbation. Columns: name, variant value, n clusters, ARI vs headline, mean centroid alignment vs headline, noise rate, and one column per primary claim (C1–C4) showing whether the claim still holds (✓ / ✗ / partial / N/A).

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
| Anchor LOO (axis-based claims) | ✓ | n/a | ✓ | n/a |
| **Sign-consistency** | **N/N ✓** | **(N-2)/N ✓ + 2 partial** | **(N-1)/N ✓** | **(N-2)/N ✓ + 2 partial** |

We pre-commit to **publishing this table whether or not it's friendly**. Sign-consistency convention: ✓ = effect same direction at p < 0.05 with effect size within ±30% of headline; ✗ = effect reverses or vanishes; partial = effect same-direction but effect size outside ±30%.

**3. Sensitivity bands on figures.** For figures reporting a primary 2024→2026 effect (F3, F5, F9, F10), shade the range across §8.2 ablations as a band around each bar / point.

**Code release.** AIES expects a public repository. We release: the reproducible Stage 4 notebook (§13.5), `figures/bertopic/` source, the prereg log, the embeddings cache, the cluster assignments parquet, and all anchor postings.

**What we do not do.** We do not show cherry-picked ablations as "robustness" while silently dropping unfriendly ones; we do not present only the median across ablations as a single number without showing the range; we do not relegate failing ablations to a footnote.

---

## 10. Risks and known limits

### 10.1 The mega-cluster problem may persist

The v3 prior found that AI/ML and traditional data-science postings collapsed into one density region. OpenAI 3072-d embeddings are denser and may resolve this — or may not. If the mega-cluster persists:

- We do **not** quietly split it with the AI-vocab regex and present the split as a found taxonomy.
- We try a hierarchical sub-clustering on the mega-cluster alone (re-run UMAP + HDBSCAN on its members) before resorting to vocab split.
- If neither resolves it, we present the mega-cluster as one family and characterize its internal vocabulary shift over time.

This is also a Stage 1 gate (§13.2): a single cluster with > 30% of postings stops Stage 1 for human decision before sub-agents fan out.

### 10.2 c-TF-IDF cross-period vocabulary turnover

If new vocabulary enters the corpus in 2026 (e.g., "agentic", "RAG"), c-TF-IDF will up-weight those terms simply because they're rare elsewhere. This artificially makes 2026-period cluster representations look more "AI-flavored" than is warranted. Mitigation: c-TF-IDF on the **pooled** corpus; per-period vocabulary diff via `topics_over_time`; report top-words alongside the diff so readers can distinguish.

### 10.3 Domain-vertical vs technical-horizontal axes confound

The v3 prior's healthcare/biotech AI cluster was a **vertical** specialization that the embedding split because the domain vocabulary dominated. We expect a similar healthcare/biotech island and possibly a fintech island. These are not "new role families" — they are domain verticals of existing role families. The §5.1 LLM naming and the §7.7 L1 crosstab must catch this; the prose must call it out.

### 10.4 LLM-naming priming

If the OpenAI naming prompt sees exemplars dominated by AI vocabulary, it labels the cluster "AI Engineer" even when only 60% of postings are AI-flavored. Mitigations:

- Naming model sees both top exemplars and 2 random members
- The alternative-label field forces the LLM to consider a second framing
- Cross-model robustness check (§7.11) flags clusters where `gpt-5.5` and `gpt-5.4-mini` disagree
- Final paper labels go through §5.2 protocol (TBD) — until then, LLM-proposed labels are working labels, and every cluster catalog carries the c-TF-IDF top-words alongside the label

### 10.5 Seed-sensitive fine structure

If §7.1 ARI < 0.4 at every K in the §4.4 sweep, individual cluster identities are not stable across seeds and per-cluster claims are weak. In that case: drop to super-family granularity (the hierarchy from §4.3), report only super-family-level claims, per-cluster claims become illustrative not load-bearing. We pre-commit that K is set by stability, not by our preference. This is a Stage 1 gate (§13.2).

### 10.6 OpenAI embedding deprecation / drift

`text-embedding-3-large` is the current model. If OpenAI silently changes the model, our results become non-reproducible. Mitigation: cache embeddings to `data/bertopic/embeddings_cache.npy` keyed by uid; never re-fetch unless the uid changes. The repo has this convention via Stage 11; we extend it.

### 10.7 Anchor-set fishing in §6 axes

The §6 embedding-space analyses depend on anchor postings, and an analyst could craft anchors to produce the result they wanted. Mitigations are in §6.6: pre-commit anchors in `config.py` with rationale, leave-one-out sensitivity, held-out validation, permutation null, anchor publication in the appendix. **The §13 sub-agent for §6 runs all four checks as part of its standard task spec, not optionally.** Failing any check downgrades the axis from headline to appendix; failing two cuts the axis entirely.

### 10.8 Sentence-embedding linearity caveat

Bolukbasi-style word-level analogy arithmetic does not transfer cleanly to sentence-level embeddings. The §6 plan accommodates this by leaning on projection (robust) rather than analogy (brittle), and the paper's prose is explicit about the distinction. Reviewers who know the literature will check this; we surface it before they ask.

### 10.9 22-month observation gap — not L3-specific

The paper's broader 22-month gap between 2024 and 2026 affects all cross-period claims, including L3. We do not solve it here; we cite the paper's gap discussion and confine L3 claims to "describes posting-frame change," not "describes employment change."

---

## 11. Pre-registration

**Frozen before fitting.** This is what we commit to.

### 11.1 Hyperparameters

§4.2 table. No changes once fit begins, except via §4.6's `min_cluster_size` sweep protocol.

### 11.2 Sample definition

§3.1 — Sample A (full SWE pool from asaniczka + arshkon + scraped, post-cap). Sample B fit two ways per §3.3. Filter rules in §2.4. Sample cap: 5 per `(company_name_canonical × period × title_normalized)`, hash-randomized within bucket.

### 11.3 K — sweep, not single value

K is **swept** across {10, 15, 20, 25, 30, 40, 50, 75} and characterized per §4.4. Headline K selected by the four pre-registered criteria in §4.4 (seed ARI ≥ 0.4, period reproduction ≥ 0.85, interpretability ≥ 3.5/5, outlier ≤ 40%). The full sweep is reported in F7 + appendix; the headline K and one super-family K go in the body.

### 11.4 Seeds

42 (primary), 1337, 2026 (stability). No silent reseeding.

### 11.5 Stopwords

§4.5 list. Frozen pre-fit. No additions during analysis.

### 11.6 Cluster naming protocol

§5.1, §5.3. **Models pinned in `config.py`:** `gpt-5.5` (snapshot `gpt-5.5-2026-04-23`) for passes < 100 calls; `gpt-5.4-mini` for passes ≥ 100 calls; no mid-pass switching. Final labeling protocol (§5.2) is TBD — until it lands, LLM-proposed labels are working labels and no paper claim depends on a label that has not been through the final protocol. Once frozen, any later change requires a paper-visible erratum note.

### 11.7 Embedding-space anchor sets

The five §6.1 axes plus the five §6.5 anchor descriptions plus the §6.4 WEAT attribute sets are committed verbatim in `figures/bertopic/config.py` before any axis is fit. The strings below are the design-doc commitment; `config.py` matches them character-for-character.

#### §6.1 axes — anchor postings (6 per pole)

**AI-native ↔ traditional**

- Positive (AI-native): "Build LLM agents and RAG pipelines."; "Fine-tune foundation models for production."; "Develop multi-agent orchestration systems."; "Build evals for LLM applications."; "Integrate vector databases for retrieval."; "Apply prompt engineering to production systems."
- Negative (traditional): "Maintain enterprise CRUD applications."; "On-call rotation for legacy services."; "Bug fixes in established codebase."; "Operate monitoring for existing systems."; "Implement features in mature framework."; "Maintain integration with internal tools."

**IC ↔ management**

- Positive (IC): "Design and implement features end-to-end."; "Write production code daily."; "Debug and ship."; "Pair-program with peers."; "Own technical decisions for a service."; "Build and ship a feature alone."
- Negative (management): "Manage a team of engineers."; "Hire and grow staff."; "Conduct performance reviews."; "Set roadmap and priorities."; "Lead 1:1s and career development."; "Drive headcount planning."

**Builder ↔ operator**

- Positive (builder): "Build new systems from scratch."; "Greenfield architecture."; "Prototype and iterate quickly."; "Ship novel features to users."; "Design new APIs."; "Architect a new platform."
- Negative (operator): "Operate and maintain existing systems."; "Reliability engineering."; "Incident response and post-mortems."; "Capacity planning for production."; "Migrate legacy infrastructure."; "Patch and harden existing services."

**Concrete ↔ abstract**

- Positive (concrete): "Implement specific feature using named tools."; "Write code in Python for a service."; "Build a recommendation component."; "Add OAuth flow to the user portal."; "Use Kubernetes for deployment."; "Ship the dashboard by end of quarter."
- Negative (abstract): "Drive technical strategy and direction."; "Evangelize engineering culture."; "Set company-wide standards."; "Define long-term technical vision."; "Influence cross-functional priorities."; "Champion best practices org-wide."

**Generalist ↔ specialist**

- Positive (generalist): "Polyglot fullstack across frontend, backend, infra."; "Comfortable across the stack."; "Wear multiple hats in a startup."; "Handle anything from UI to database."; "Ship across product surfaces."; "Cross-domain delivery."
- Negative (specialist): "Deep expert in Kubernetes."; "Specialist in distributed databases."; "10+ years on the iOS platform."; "Recognized authority on COBOL mainframe."; "Focus exclusively on the search infrastructure."; "Master of CUDA optimization."

#### §6.5 anchor descriptions

- AI Engineer: "A SWE who builds RAG agents and integrates LLMs into production."
- Frontend Engineer: "A SWE who builds React UIs and ships features to web users."
- Backend Engineer: "A SWE who designs APIs and operates server infrastructure."
- Legacy Specialist: "A SWE who maintains a COBOL mainframe codebase."
- SRE: "A SWE who runs reliability for production systems."

#### §6.4 WEAT attribute sets

Each set is 6 short anchor sentences, embedded individually and averaged.

- **Innovation:** "Pioneer new technology."; "Prototype novel ideas."; "Push the state of the art."; "Build greenfield products."; "Explore new methods."; "Pioneer ML applications."
- **Maintenance:** "Keep existing systems running."; "Patch and stabilize."; "Reduce technical debt."; "Operate at high reliability."; "Maintain backward compatibility."; "Sustain mature systems."
- **Orchestration + mentorship:** "Decompose work for engineers."; "Author specs and ADRs."; "Mentor junior engineers."; "Onboard new team members."; "Pair-program for knowledge transfer."; "Lead through influence."
- **Solo IC:** "Ship features individually."; "Own a service end-to-end."; "Code without supervision."; "Independently debug production issues."; "Self-direct daily work."; "Single-author technical decisions."
- **Architecture + design:** "Design system architecture."; "Author technical RFCs."; "Decompose monoliths."; "Design APIs for scale."; "Plan platform evolution."; "Architect cross-service flows."
- **Implementation:** "Write production code."; "Implement well-defined tickets."; "Add features to an existing module."; "Refactor specified code."; "Ship implementation tasks."; "Code against a written spec."
- **Growth:** "Scale the platform."; "Expand customer base."; "Drive product growth."; "Launch new markets."; "Grow capacity."; "Increase throughput."
- **Stability:** "Maintain existing customers."; "Preserve uptime."; "Reduce churn."; "Stabilize releases."; "Sustain SLAs."; "Lock in current revenue."
- **Exploration:** "Investigate new approaches."; "Experiment with techniques."; "Pilot novel architectures."; "Explore new vendors."; "Try new tools."; "Investigate the unknown."
- **Exploitation:** "Optimize known approach."; "Tune existing systems."; "Refine current methods."; "Squeeze performance from production code."; "Improve known metrics."; "Maximize current returns."

### 11.8 Embedding-space pre-registration discipline

§6.6. Anchor leave-one-out, held-out validation, permutation null. Run by the §13 sub-agent for §6 as a non-optional part of the spec. Failing any check downgrades the axis; failing two cuts it.

### 11.9 Validation thresholds — stated in advance

| Check | Threshold | If violated |
|---|---|---|
| Per-period reproduction centroid cosine | ≥ 0.85 | Cross-period claims demoted to within-cluster only |
| Seed-pair ARI at headline K | ≥ 0.4 | Per-cluster claims demoted; super-family-level only |
| Within-2024 cross-source alignment (§7.6) | ≥ 0.85 centroid cosine | Cross-period effects flagged as partly source-confounded |
| Outlier rate (HDBSCAN) | ≤ 40% | If higher, retune `min_cluster_size` once via §4.6, document in `prereg_log.md` |
| Largest cluster's posting share | ≤ 30% | Stage 1 stops; mega-cluster gate per §13.2 |
| §6 axis leave-one-out cosine spread | ≤ 0.10 | Axis flagged as unstable; reported with caveat |
| §6 held-out anchor hit rate | ≥ 80% | Axis reported with caveat or cut |
| §6.4 WEAT Cohen's d | ≥ 0.5 with Bonferroni p < 0.01 | Below threshold reported as null, not cut |

### 11.10 Things we will not do

- Hand-tune UMAP / HDBSCAN / axes / anchors to make a specific cluster or finding appear or disappear
- Choose headline K to maximize a specific finding's effect size (selected by §4.4 criteria, full)
- Drop seeds whose ARI is inconvenient
- Rename clusters to support claims after labels are frozen
- Use `reduce_outliers` and report the post-reduction noise rate as the only number
- Add stopwords mid-analysis to make a cluster's top-words read more cleanly
- Re-run with a different embedding model and silently switch headlines
- Drop §6 axes that produced inconvenient signals; report the leave-one-out and held-out checks even if they fail
- Add anchors mid-analysis without an erratum entry in `prereg_log.md`

### 11.11 Paper-visible audit trail

A pre-registration log (`figures/bertopic/prereg_log.md`) records: timestamp of this doc's commit, hyperparameter freeze timestamp, anchor freeze timestamp, Stage 1.5 mini-gate outcomes, Stage 2 task IDs and timestamps, Stage 3 cull decisions with rationale, and any deviation from this doc with rationale. The log is committed to source control and goes in the paper appendix.

---

## 12. Compute and runtime estimate

| Step | Time (rough) | Memory |
|---|---|---|
| Stage 0 — embedding cache build (postings + anchors) | < 5 min | 1 GB |
| Stage 0 — pre-flight checks + smoke test on 5% | < 10 min | 1 GB |
| Stage 1 — `min_cluster_size` sweep (5 values) | 15–20 min | 2 GB |
| Stage 1 — headline fit + K sweep (8 values) | 10–15 min | 2 GB |
| Stage 1 — seed stability (3 seeds × full pipeline) | 20–30 min | 2 GB |
| Stage 1 — LLM naming (~50 calls @ gpt-5.5) | 2–3 min, ~$2 | trivial |
| Stage 1 — determinism + mega-cluster gates | < 1 min | trivial |
| Stage 2 sub-agents (parallel; per task) | 30–90 min | up to 3 GB each |
| Stage 2 axis projection (§6.1) | 30 min | 2 GB |
| Stage 2 boundary (§6.2) | 15 min | 1 GB |
| Stage 2 centroid drift (§6.3) | 30 min | 2 GB |
| Stage 2 WEAT (§6.4) | 30 min, ~$1 | 1 GB |
| Stage 2 anchor neighborhood (§6.5) | 15 min | 1 GB |
| Stage 2 stability suite (§7) | 60–90 min | 3 GB |
| Stage 2 secondary ablations (§8.2) | 2–3 hr total | up to 3 GB |
| **End-to-end with caching** | **Stage 0+1: ~1 hr; Stage 2: ~3 hr wall (parallel); Stage 3+4: ~1 hr human** | **peak ~3 GB** |

Well within the 31 GB constraint. Caches: UMAP output, HDBSCAN output, c-TF-IDF matrix, LLM labels — all keyed on the sample's uid hash, separated by sample.

---

## 13. Implementation plan and sub-agent delegation

The work runs in five stages: Stage 0 infrastructure → Stage 1 core BERTopic → Stage 1.5 mini-gate → Stage 2 parallel sub-agent fan-out → Stage 3 critical-evaluation cull → Stage 4 reproducible notebook. The orchestrator (a Claude Code session driven by `figures/bertopic/orchestrator_prompt.md`) executes Stages 0 and 1 sequentially, stops at Stage 1.5 for human sign-off, fans Stage 2 out to sub-agents in parallel, and produces the Stage 3 cull recommendations. Stage 4 is a separate session driven by humans against the surviving artifacts.

### 13.1 Stage 0 — Infrastructure

Sequential, foundational. Nothing downstream begins until Stage 0 commits.

- **S0.1 — `config.py` skeleton.** Build `figures/bertopic/config.py` with: hyperparameters (§4.2); LLM model IDs and snapshot pins (§5.1); sample-cap rules (§3.1); seeds (§11.4); paths (`PROJECT_ROOT`-relative); anchor sets verbatim from §11.7. PR-tracked. The single source of truth.
- **S0.2 — `sample.py`.** DuckDB-driven Sample A and Sample B build. Outputs `intermediate/sample_a.parquet`, `intermediate/sample_b.parquet`, `intermediate/sample_sizes.csv`. Logs the row count per source × period.
- **S0.3 — Embedding cache build.** Read 3072-d posting embeddings from `unified_core.parquet`, embed all anchor postings (§11.7) with `text-embedding-3-large`, save as a single numpy file at `data/bertopic/embeddings_cache.npy` with a sidecar `embeddings_cache.index.parquet` mapping `(uid_or_anchor_id) → row_index`. Anchors and postings share the same array and one I/O path.
- **S0.4 — Pre-flight checks.** All required columns present in `unified_core.parquet`; embedding dimension == 3072; L2 norms in [0.99, 1.01]; no nulls where there shouldn't be; sample sizes within ±20% of expected. Fail loud on any mismatch.
- **S0.5 — Smoke test.** Run the full pipeline (UMAP → HDBSCAN → c-TF-IDF → 1 LLM naming call) on a 5% stratified slice of Sample A. Five-minute run, catches BERTopic version drift, OpenAI auth, slow steps, obvious mega-cluster red flags. The Stage 1 sweep does not start until smoke passes.

### 13.2 Stage 1 — Core BERTopic

Sequential. The orchestrator runs this stage; sub-agents do not. Outputs are hash-anchored at the end so Stage 2 sub-agents can verify they're working off the right baseline.

- **S1.1 — `min_cluster_size` sweep.** §4.6 protocol on Sample A. Output: `intermediate/mcs_sweep.csv`.
- **S1.2 — Headline fit + K sweep.** §4.4 protocol with the chosen `min_cluster_size`. Output: `intermediate/k_sweep.csv`, `intermediate/raw_fit.bertopic`.
- **S1.3 — Seed stability gate (§7.1).** Three seeds × headline K. **Gate:** if any pair has ARI < 0.4 AND centroid alignment < 0.85, stop and surface for human decision; do not fan out Stage 2 on an unstable foundation.
- **S1.4 — Mega-cluster gate (§10.1).** Compute the largest cluster's posting share. **Gate:** if > 30%, stop. Try hierarchical sub-clustering on the mega-cluster and report. Do not let a silent mega-cluster propagate to Stage 2.
- **S1.5 — Determinism check.** Run S1.2 a second time with the same seed and config. Compare BERTopic model output hashes (HDBSCAN labels, UMAP coordinates). They must be byte-identical. If not, find the source of nondeterminism — usually `n_jobs > 1` in UMAP or HDBSCAN — fix it, re-run S1.2 and S1.3 from scratch.
- **S1.6 — LLM naming (§5.1).** `gpt-5.5` proposes labels for headline-K clusters and super-family-K clusters. Output: `data/bertopic/topic_info.parquet` (pre-Stage 2 form: includes `gpt55_label`, awaits `gpt54mini_label` from §7.11 in Stage 2).
- **S1.7 — Hash artifacts + tag commit.** Compute `model_hash`, `sample_hash`, `config_hash` (SHA256 of file contents). Write to `intermediate/stage1_freeze.json`. Commit with git tag `stage1-freeze-<date>`. This is the hash bundle every Stage 2 sub-agent will verify before running.

### 13.3 Stage 1.5 — Mini-gate (human)

The orchestrator produces `figures/bertopic/memos/stage1_freeze.md` containing:

1. Headline K and the sweep curve, with the four §4.4 criteria scored
2. Headline `min_cluster_size` and the sweep table
3. Noise rate (raw HDBSCAN, before reduce_outliers)
4. Seed-pair ARI at headline K (3 seeds)
5. Mega-cluster check: largest cluster's posting share
6. AI-region structure: ≥ 1 AI-flavored cluster? Differentiated or one mega? (Pre-feeds §1.4.4)
7. Determinism: was the double-run identical?
8. Flags for Stage 2: anything sub-agents need to know
9. Stage 1 cluster catalog: per cluster — c-TF-IDF top-10, KeyBERTInspired top-5, posting count, period split, top 3 firms, top 3 metros

The orchestrator stops here and waits for author sign-off. Sign-off updates `prereg_log.md` and unblocks Stage 2.

### 13.4 Stage 2 — Parallel sub-agent fan-out

After mini-gate sign-off, the orchestrator launches eight sub-agent tasks in parallel.

#### Sub-agent execution standard

Every Stage 2 sub-agent invocation:

- **Model:** `claude-opus-4-7` (Opus 4.7) — most capable; the analyses are non-trivial and justify the cost.
- **Effort:** **high** — verbose reasoning, no early termination, full robustness suite per the per-task spec. The sub-agent runs every check the spec lists, even if early evidence looks compelling.
- **Subagent type:** `general-purpose`.
- **Time budget:** per task. If a sub-agent exceeds 2× budget, it pauses and asks the orchestrator for guidance rather than silently finishing partial work.
- **Prompt:** self-contained per the per-task spec, with hash-anchored Stage 1 inputs (`model_hash`, `sample_hash`, `config_hash`). The sub-agent's first action is verifying these hashes match the frozen Stage 1 outputs. If they don't match, the sub-agent fails loud.
- **Read-before-running:** every sub-agent reads the relevant sections of `figures/bertopic/design.md` plus `figures/bertopic/config.py` before touching code. The spec lists which sections.
- **Memo discipline:** every sub-agent ends with `figures/bertopic/memos/<task-id>.md` containing: (a) what was run with exact parameters, (b) tables/figures produced with paths, (c) the three-gate evaluation per §13.5, (d) `recommend_for_paper: yes / no / conditional` with one-paragraph rationale.
- **No advocacy:** sub-agents do not advocate for inclusion of their own work. The memo's job is to give the orchestrator and authors what's needed to decide, not to defend the work.
- **No silent retries:** failures are surfaced to the orchestrator, not papered over.

#### Stage 2 task list

| Task ID | Sections | Time budget | Output | Memo path |
|---|---|---|---|---|
| **T-bootstrap** | §7.2 bootstrap, §7.3 per-period, §7.6 within-2024 cross-source | 90 min | `data/bertopic/stability.parquet`, T2 numbers | `memos/t_bootstrap.md` |
| **T-method** | §7.4 NMF baseline, §7.5 MiniLM cross-embedding | 60 min | T3 numbers | `memos/t_method.md` |
| **T-quality** | §7.8 coherence + diversity, §7.9 silhouette + size dist, §7.10 noise rate, §7.11 cross-model naming | 45 min | Quality block of T2/T3 | `memos/t_quality.md` |
| **T-axis** | §6.1 — five axes, projection per posting, period-mean shift, anchor sensitivity | 45 min | `data/bertopic/axes.parquet`, `axis_projections.parquet`, `cluster_axis_profile.parquet`, F9 candidate, T7 | `memos/t_axis.md` |
| **T-boundary** | §6.2 — boundary fraction over cluster pairs, change 2024→2026 | 30 min | `data/bertopic/boundary_postings.parquet` | `memos/t_boundary.md` |
| **T-drift** | §6.3 — per-cluster centroid drift vector, control-differenced, axis decomposition | 45 min | `data/bertopic/centroid_drift.parquet`, T8 | `memos/t_drift.md` |
| **T-weat** | §6.4 — 5 pre-registered tests, Cohen's d + permutation p + Bonferroni | 30 min | `data/bertopic/weat_results.parquet`, T9 | `memos/t_weat.md` |
| **T-anchor** | §6.5 — neighborhood diffusion at 4 cosine thresholds | 30 min | `data/bertopic/anchor_neighborhoods.parquet`, F10 candidate | `memos/t_anchor.md` |
| **T-l1l2** | §7.7 read-only crosstab (gated on L1/L2 columns existing in `unified_core.parquet`) | 30 min | `data/bertopic/l1_l2_crosstab.parquet` | `memos/t_l1l2.md` |
| **T-ablations** | §8.2 secondary ablations | 3 hr | `data/bertopic/ablations.parquet`, T4, T6 | `memos/t_ablations.md` |

The orchestrator launches T-bootstrap, T-method, T-quality, T-axis, T-boundary, T-drift, T-weat, T-anchor in parallel. T-l1l2 runs in parallel if and only if `role_family_l1` and `skill_theme_*` columns exist in `unified_core.parquet`; otherwise it is queued for a later session. T-ablations runs after the other Stage 2 tasks complete (it depends on their cluster-level outputs for the T6 sign-consistency matrix).

#### Verification by orchestrator

When a sub-agent reports back, the orchestrator:

1. Reads the memo end-to-end.
2. Spot-checks **at least one** quantitative claim per memo by opening the artifact and verifying the number matches the memo. Bad numbers in memos are a sign of a bigger problem.
3. Verifies the methodology section of the memo matches the design doc's spec. Sub-agents that deviated without justification get re-tasked.
4. Looks for advocacy language (memos that read as defending the analysis rather than reporting it). Flag for stricter review.
5. Records verification in `prereg_log.md`: task ID, hash bundle verified, claims spot-checked, deviations from spec.

### 13.5 Stage 3 — Critical-evaluation cull

The orchestrator produces `figures/bertopic/memos/synthesis.md` — the document the user reads. Each Stage 2 finding goes through three gates. **All three must pass for paper inclusion.** Failing one → "reported as exploratory, framed accordingly in prose." Failing two → cut.

#### Gate 1 — Narrative

The finding is explainable in 2–3 sentences. It supports a named paper claim (C1–C4 or T1–T4 from §1.4). It would survive a hostile reviewer asking "what does this actually show?"

#### Gate 2 — Effect size

Per analysis:

| Analysis | Threshold |
|---|---|
| Axis projection | Period-mean shift on axis ≥ 0.05 cosine units AND ≥ 3× the leave-one-out anchor sensitivity |
| Boundary postings | Boundary-fraction change ≥ 5pp AND permutation p < 0.05 |
| Centroid drift | Drift magnitude ≥ 2× control-occupation drift on the same axis |
| WEAT | Cohen's d ≥ 0.5 AND Bonferroni-corrected p < 0.01 across all reported tests |
| Anchor neighborhood | Trend monotonic across all of {0.5, 0.6, 0.7, 0.8} cosine thresholds |
| BERTopic claims | C1–C4 thresholds in §1.4.1; T6 sign-consistency at p < 0.05 with effect size within ±30% of headline |

#### Gate 3 — Robustness

The finding survives at least 3 of:

- Seed reshuffle (for stochastic steps)
- Anchor leave-one-out (for axis-based analyses)
- Subset replication on at least one source / period / seniority slice
- Permutation null (sign-consistency at p < 0.05)
- Cross-embedding (MiniLM) replication

#### Synthesis structure

The orchestrator's `synthesis.md` is **prose, not bullet lists**. Under 2,000 words. Structure:

1. **Executive summary** — three sentences on what survived, what didn't, what the paper narrative should commit to.
2. **Survivors** — per finding, in priority order: claim it supports (C1, C2, C3, C4, T1–T4, §1.4.3, §1.4.4); headline number / figure with path; three-gate evaluation; one-paragraph proposed prose in The Economist register per `AGENTS.md`.
3. **Cuts** — per finding dropped: which gate it failed; whether worth retrying with a different design.
4. **Lessons** — observations about the data, methodology, or analysis the human authors should know: mega-cluster behavior; embedding-model surprises; anchor sensitivity; design-doc mistakes that should be revised before the next iteration.
5. **Recommendations for the paper** — which findings are headline-worthy, appendix-worthy, or drop.
6. **Open questions for human work** — what the authors must do next (the §5.2 reviewer protocol, anchor-set domain validation, etc.).

### 13.6 Stage 4 — Reproducible notebook (publishable artifact)

A separate human session, after Stage 3 sign-off. A single IPython notebook, `figures/bertopic/bertopic_paper.ipynb`, that:

- Imports from `figures/bertopic/config.py` (frozen hyperparameters, model IDs, anchor sets, paths)
- Loads `data/unified_core.parquet` and `data/bertopic/embeddings_cache.npy`
- Builds Sample A and Sample B from the frozen rules
- Refits BERTopic from cached embeddings (no re-encoding)
- Reproduces every **surviving** figure and table from §9 — only the artifacts that passed §13.5's three gates
- Runs end-to-end in < 30 minutes on the cached path; clearly flags which cells are slow / require API access

The exploratory scripts in Stage 1 and Stage 2 stay in the repo for audit but are not the headline artifact.

### 13.7 Code structure

```
figures/bertopic/
  __init__.py
  config.py                       # frozen hyperparameters, model IDs, anchor sets, paths
  sample.py                       # build Sample A, Sample B (DuckDB + pyarrow)
  embedding_cache.py              # S0.3 build + load
  preflight.py                    # S0.4 checks
  smoke_test.py                   # S0.5
  stage1/
    mcs_sweep.py                  # S1.1
    fit.py                        # S1.2
    seed_stability.py             # S1.3
    mega_cluster_gate.py          # S1.4
    determinism_check.py          # S1.5
    naming.py                     # S1.6
    freeze.py                     # S1.7
  stage2/
    t_bootstrap.py                # §7.2, 7.3, 7.6
    t_method.py                   # §7.4, 7.5
    t_quality.py                  # §7.8, 7.9, 7.10, 7.11
    t_axis.py                     # §6.1
    t_boundary.py                 # §6.2
    t_drift.py                    # §6.3
    t_weat.py                     # §6.4
    t_anchor.py                   # §6.5
    t_l1l2.py                     # §7.7 (gated)
    t_ablations.py                # §8.2
  artifacts.py                    # write all parquets per §9.1
  intermediate/                   # Stage-0/1 scratch outputs (gitignored)
  memos/                          # Stage-1 freeze + Stage-2 task memos + synthesis (committed)
  fig_*.py                        # one file per F1–F10
  tab_*.py                        # one file per T1–T9
  bertopic_paper.ipynb            # Stage-4 reproducible notebook (the release artifact)
  prereg_log.md                   # §11.11 audit trail, copied to paper appendix
  orchestrator_prompt.md          # §13's orchestrator prompt — drives the whole project
  design.md                       # this document
```

`config.py` is the single source of truth for hyperparameters, LLM model pins, anchor sets, and sample-cap rules; everything else imports from it. The reproducible notebook (`bertopic_paper.ipynb`) imports from both `stage1/` and `stage2/` but does not redefine any hyperparameter.

### 13.8 Code style and quality bar

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
- **Tooling.** `ruff` formats and lints; the repo's config is the only opinion. `mypy --strict` clean on `figures/bertopic/` before Stage 1.5 mini-gate.
- **Boy Scout.** Leave nearby code cleaner than you found it. If you understand a messy bit while passing through, fix it in the same commit.

If a rule above conflicts with the existing repo's conventions, the repo wins — consistency over local optimization.

---

## 14. Reproducibility and code release

AIES expects an accessible reproduction path. Our release plan:

| Artifact | Released | Location |
|---|---|---|
| Stage 4 notebook (`bertopic_paper.ipynb`) | Yes | Public repo |
| `figures/bertopic/` source (Stage 0/1/2 scripts + Stage 4 helpers) | Yes | Public repo |
| `config.py` (frozen hyperparameters, model versions, anchor sets) | Yes | Public repo |
| Pre-registration log (`prereg_log.md`) | Yes | Public repo |
| Orchestrator prompt (`orchestrator_prompt.md`) | Yes | Public repo |
| Cluster assignments (`assignments.parquet`) | Yes | Public repo |
| Cluster labels and naming proposal record (T5) | Yes | Public repo |
| Anchor postings and §6 axes | Yes — verbatim in §11 of this doc and in `config.py` | Public repo |
| Embedding cache, uid- and anchor-id-keyed | Yes — large file via Git LFS or a separate release tarball | Public repo / release |
| `unified_core.parquet` slice | Yes — only fields used by the notebook | Public repo |
| Raw scraped HTML / pre-pipeline data | No — covered by paper's data-availability section |
| LLM-naming raw API responses | Yes, sanitized | Public repo |

**Pinned dependencies.** `requirements.txt` for the BERTopic release pins exact versions of: `bertopic`, `umap-learn`, `hdbscan`, `sentence-transformers`, `scikit-learn`, `gensim`, `openai`, `pyarrow`, `duckdb`, `numpy`, `scipy`. A `Dockerfile` is nice-to-have, not required.

**Embedding-model versioning.** OpenAI embeddings are not deterministic across silent server-side updates (§10.6). The cache is the canonical artifact: any re-encoding produces a number that may differ from the paper's. `config.py` records the exact OpenAI model string and the date of the cache build; `prereg_log.md` records the cache hash.

**Ethics statement (L3-specific).** Three concerns and our mitigations:

- **LLM cluster-name bias.** `gpt-5.5` and `gpt-5.4-mini` may name clusters using stereotyped or marketing-flavored vocabulary inherited from training data. Mitigations: (a) c-TF-IDF top-words reported in every cluster catalog alongside the LLM-proposed label; (b) cross-model proposal sanity check (§7.11); (c) final paper labels go through §5.2 protocol (TBD).
- **Anchor-set bias in §6.** The anchors are author-written. We publish every anchor verbatim (§11.7) so readers can read what shaped each axis. Anchor leave-one-out and held-out validation (§6.6) catch axis instability.
- **Public-data posture.** The cluster assignments and axis projections are derived from public LinkedIn job postings. We release uid-keyed assignments without re-publishing posting text in the release tarball — anyone who wants posting text fetches it via the upstream pipeline.

**End of design doc.**
