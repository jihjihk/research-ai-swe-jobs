# T-quality — coherence, silhouette, noise rate, cross-model naming on the headline K=10 fit

## What was run

Stage 2 sub-agent driver: `figures/bertopic/stage2/t_quality.py` (single
file, no shared utils). Reads only frozen Stage 1 artifacts; does not
refit BERTopic. End-to-end runtime ~7 minutes (dominated by gensim C_v
on 57,766 tokenised Sample A documents and 9 OpenAI naming calls).

Hashes verified on first action (all five match `stage1_freeze.json`):

| Artifact | Path | sha256 |
|---|---|---|
| model | `figures/bertopic/intermediate/raw_fit.bertopic` | `d51f15e6…509415` |
| sample | `figures/bertopic/intermediate/sample_a.parquet` | `6719a025…feb265` |
| embeddings | `data/bertopic/embeddings_cache.npy` | `29d77bf9…b479b27` |
| assignments | `data/bertopic/assignments.parquet` | `a03bc515…808ab82` |
| config | `stage1_freeze.json: config_hash` | `bef20ab2…07bce8` |

Note: the spec writes the model path as `data/bertopic/model.bertopic`,
but the on-disk path is `figures/bertopic/intermediate/raw_fit.bertopic`
(`config.RAW_FIT_PATH`); the sha256 of that file matches the frozen
`model_hash` exactly, so we proceed against it.

Per-section parameters:

- **§7.10 noise rate.** Loaded the raw fit, called
  `model.reduce_topics(docs, nr_topics=10)`, recorded the noise rate of
  the resulting `topics_`, then called
  `model.reduce_outliers(docs, topics, strategy='embeddings',
  embeddings=postings_3072d)` and recorded the new noise rate. Postings
  embeddings pulled from `embeddings_cache.npy` via the index
  (`kind='posting'`).
- **§7.9 silhouette + size.** Loaded `model.umap_model.embedding_`
  (shape `(57766, 5)`, the same 5-D UMAP space that fed HDBSCAN, seeded
  with `random_state=42`); ran `sklearn.metrics.silhouette_samples` with
  Euclidean metric on the 39,622 non-outlier rows using the headline
  K=10 cluster ids from `assignments.parquet`. Cluster-size statistics
  computed over the same 9 non-outlier clusters.
- **§7.8 coherence + diversity.** Reference corpus = Sample A
  `description_core_llm` tokenised with `config.TOKEN_PATTERN`
  (lowercased). Topic word lists = top-10 c-TF-IDF terms from
  `topic_info.parquet`. Multi-word phrases were split into unigram
  tokens and filtered to dictionary tokens, capped at 10 per topic; all
  9 topics retained ≥ 2 unigrams. Coherence via
  `gensim.models.CoherenceModel` with `coherence ∈ {c_npmi, u_mass,
  c_v}` and `topn=10`. Topic diversity uses the *original* phrase-level
  top-10 lists, lowercased, per Dieng et al. 2020.
- **§7.11 cross-model naming.** Re-ran the §5.1 protocol with
  `figures.bertopic.stage1.naming.propose_label(model="gpt-5.4-mini")`
  using the identical exemplar selection (top-15 c-TF-IDF terms,
  5 representative + 2 random `seed=cid` exemplars with first-200-char
  snippets). All 9 cluster names returned; no LLM-call failures. Labels
  embedded in a single batch via `text-embedding-3-large` (one batched
  POST to `/v1/embeddings`). Exact match is case-insensitive,
  whitespace-collapsed.

Outputs written:

- `data/bertopic/topic_quality.parquet` — 9 rows, columns
  `(topic_id, n_members, npmi, umass, c_v, silhouette)`
- `data/bertopic/topic_info_with_naming.parquet` — 9 rows, original
  topic_info columns + `gpt54mini_label / gpt54mini_confidence /
  gpt54mini_alternative / exact_match / label_cosine`

## Results

### §7.10 honest noise rate

| Stage | n_outliers | noise_rate |
|---|---:|---:|
| HDBSCAN raw / headline K=10 (before) | 18,144 / 57,766 | **0.314** |
| After `reduce_outliers(strategy='embeddings')` | 0 / 57,766 | **0.000** |

The "after" rate is 0.0 because the embeddings strategy reassigns every
-1 document to its nearest non-outlier topic in 3072-d space — by
construction. The 31.4% figure is the honest number; the 0% number
exists only to demonstrate what `reduce_outliers` can hide.

### §7.9 silhouette + cluster sizes

Overall mean silhouette (5-D UMAP, Euclidean, n=39,622 non-outlier
points) = **0.224**. Below the §7.9 0.4 strong-separation marker.

Per-cluster silhouette (sorted by topic_id):

| topic_id | gpt-5.5 label | n | silhouette |
|---:|---|---:|---:|
| 0 | AI Software Engineering | 15,055 | **−0.119** |
| 1 | Test Automation Engineer | 7,843 | 0.454 |
| 2 | Data Engineer | 6,782 | 0.614 |
| 3 | Salesforce Cloud Developer | 4,232 | **−0.213** |
| 4 | Full Stack Developer | 2,807 | 0.514 |
| 5 | Mobile Application Developer | 1,058 | 0.892 |
| 6 | E-commerce Software Engineering | 856 | 0.859 |
| 7 | Application Systems Analyst | 607 | 0.534 |
| 8 | ServiceNow Developer | 382 | 0.980 |

Two of the largest three clusters (AI Software Engineering, Salesforce
Cloud Developer) are negative — i.e., the average member is closer to
some other cluster's centroid than to its own in 5-D UMAP space. The
small specialist clusters (Mobile, E-commerce, ServiceNow) are tightly
separated.

Cluster-size distribution at headline K=10 (excluding -1):
n=9 clusters, median 2,807, IQR [856, 6,782], p5/p95 = [472, 12,170],
min 382, max 15,055. Largest cluster share = 26.06% (AI Software
Engineering) — under the §13.2 30% mega-cluster gate, consistent with
the freeze JSON.

### §7.8 coherence + diversity

Aggregate (n=9 topics):

| Metric | Mean | Median | §7.8 target | Pass? |
|---|---:|---:|---|---|
| NPMI (c_npmi) | **0.044** | 0.028 | mean ≥ 0.05 | borderline (mean 0.044 < 0.05) |
| UMass | −1.282 | — | closer to 0 better | (no target) |
| C_v | **0.480** | 0.458 | mean ≥ 0.45 | **pass** |
| Topic diversity | **0.833** | — | ≥ 0.6 | **pass** |

Per-topic detail in `data/bertopic/topic_quality.parquet`. The mobile
(c5) and e-commerce (c6) clusters are the most coherent (c_v 0.667 and
0.520; npmi 0.180 and 0.024 respectively). The AI Software Engineering
mega-cluster (c0) has the lowest coherence (npmi 0.014, c_v 0.385),
consistent with its negative silhouette and 26% size share — its top
words `[ai, software engineering, automation, engineers, software
development, engineer, expertise, engineering, devops, architecture]`
are a generic AI-engineering bag, not a tightly co-occurring vocabulary.

### §7.11 cross-model naming (gpt-5.4-mini vs gpt-5.5 primary)

| topic_id | gpt-5.5 (primary) | gpt-5.4-mini (secondary) | exact | cosine |
|---:|---|---|:---:|---:|
| 0 | AI Software Engineering | AI Engineering | no | 0.812 |
| 1 | Test Automation Engineer | Software Engineer | no | 0.598 |
| 2 | Data Engineer | Data Engineer | **yes** | 1.000 |
| 3 | Salesforce Cloud Developer | Salesforce Developer | no | 0.908 |
| 4 | Full Stack Developer | Full Stack Developer | **yes** | 1.000 |
| 5 | Mobile Application Developer | Mobile App Developer | no | 0.923 |
| 6 | E-commerce Software Engineering | TikTok E-commerce | no | **0.476** |
| 7 | Application Systems Analyst | Application Analyst | no | 0.785 |
| 8 | ServiceNow Developer | ServiceNow Developer | **yes** | 1.000 |

Summary: **exact-match rate 33.3% (3/9)**, **mean cosine 0.834**,
median 0.908, min 0.476 (cluster 6, where the mini fixates on
"TikTok"), share with cosine ≥ 0.85 = **55.6% (5/9)**.

The §7.11 "sanity floor" is exact-match ≥ 50% **or** mean cosine
≥ 0.85. Exact-match falls below 50%, and mean cosine falls just below
0.85. Per the design doc: "If exact-match rate < 50% or mean cosine <
0.85, the LLM-proposed labels are model-sensitive and we surface this
in the paper alongside the cluster catalog." This run hits both
trigger conditions, so the labels should be presented as
model-sensitive in the paper. One cluster (c1: Test Automation vs
Software Engineer) is a substantive disagreement, not a paraphrase —
the §5.2 human-review protocol should adjudicate.

## Three-gate evaluation (per design.md §13.5)

This task is a **diagnostic block**, not a finding that backs a
specific C/T claim — its job is to populate the §7.8/§7.9/§7.10/§7.11
quality numbers that go into Table T2/T3, and to flag whether the
headline-K cluster catalog is fit for paper inclusion.

- **Gate 1 (Narrative).** **Conditional.** No direct C1–C4 / T1–T4
  support. Indirectly bears on C1 (the AI Software Engineering cluster
  exists, has a 26% mass share, but is the *least* coherent and least
  separated of the nine — consistent with §1.4.4 "AI Engineer is named
  but not yet differentiated geometrically"). Also a methodological
  signal: the headline taxonomy has internal heterogeneity that the
  paper's prose should acknowledge.
- **Gate 2 (Effect size).** Mixed. C_v mean 0.480 ≥ 0.45 target
  (pass); diversity 0.833 ≥ 0.6 (pass); NPMI mean 0.044 vs ≥ 0.05
  target (fail by 0.006); silhouette overall 0.224 vs ≥ 0.4 marker
  (fail); cross-model exact-match 33.3% (fail vs 50%) and cosine 0.834
  (fail vs 0.85). The largest cluster's silhouette is negative.
- **Gate 3 (Robustness).** Not applicable as a single check — quality
  metrics are descriptive of the frozen fit, not a finding to be
  perturbed. Cross-references:
  - Seed reshuffle: the §7.1 freeze recorded mean pairwise seed ARI
    0.699 at headline K, so the cluster *labels* are not seed-stable
    enough to interpret silhouette to two decimals across seeds.
  - Cross-embedding: §7.5 (T-method, separate sub-agent) will rerun on
    MiniLM and provide the cross-embedding signal for these numbers.
  - Permutation null: not run for coherence; topic diversity has a
    well-known size dependence so permutation isn't informative here.
  Gate 3 status: **partial** — the underlying fit's stability is
  borderline (per Stage 1), so any per-cluster quality number should
  be reported with a caveat.

## recommend_for_paper: conditional

## Rationale

The four blocks are diagnostic: they belong in the paper's Table T2
(quality block) and T3 (cross-model naming) regardless of whether the
numbers are flattering. The numbers themselves are mixed: C_v and
topic diversity clear their §7.8 targets; NPMI misses by 0.006; the
overall silhouette (0.224) is well below the §7.9 strong-separation
marker and is *negative* for the two largest clusters (AI Software
Engineering and Salesforce Cloud Developer); the gpt-5.4-mini
cross-naming hits both §7.11 trigger conditions (exact-match 33.3%,
mean cosine 0.834), which the design doc says obliges us to surface
the model-sensitivity in the paper alongside the cluster catalog.
Recommend including the Table T2 / T3 numbers verbatim and presenting
the AI Software Engineering cluster as a coarse aggregate rather than
a cleanly-separated role family — this matches the §1.4.4 prior that
the geometry has not yet differentiated AI sub-roles. Whether the
paper retains the §5.1 LLM-proposed labels as the working catalog or
defers to §5.2 human review is a §5.2 decision, not this memo's.
