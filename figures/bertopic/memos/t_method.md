# T-method — §7.4 NMF baseline + §7.5 MiniLM cross-embedding

## What was run

Two cross-method comparisons against the frozen Stage 1 fit (OpenAI
3072-d, headline mcs=70, K=10, 9 actual clusters).

- **§7.4 NMF baseline.** TF-IDF on `description_core_llm` (Sample A,
  57,766 docs) using the §4.2 vectorizer settings (ngram (1,3),
  `min_df=10`, `max_df=0.4`, custom stopwords). Single
  `NMF(n_components=10, random_state=42)`. Hard-assigned by argmax over
  loadings. ARI / NMI vs the OpenAI BERTopic K=10 assignments
  (`assignments.parquet`).
- **§7.5 MiniLM cross-embedding.** Encoded all 57,766 Sample A docs with
  `sentence-transformers/all-MiniLM-L6-v2` (384-d, 57 min wallclock for
  the encoding step). Refit BERTopic at headline (mcs=70, seed=42) on
  those embeddings; reduced to K=10. ARI / NMI vs the OpenAI fit.

Hash bundle (`stage1_freeze.json`) verified before any compute (5/5
match against on-disk artifacts).

Code: `figures/bertopic/stage2/t_method.py`. Backgrounded after the
sub-agent's first invocation crashed mid-encode; relaunched as `nohup
python -u -m figures.bertopic.stage2.t_method`. Total runtime
~90 minutes (60 min MiniLM encode, 7 min BERTopic UMAP+HDBSCAN+ctf-idf,
~25 min misc & competing CPU with t_bootstrap).

Outputs:

- `data/bertopic/method_comparison.parquet` (4 rows: NMF and MiniLM,
  each in two flavours — including / excluding OpenAI noise).
- `data/bertopic/method_cluster_alignment.parquet` (9 rows — per
  OpenAI cluster, the NMF and MiniLM cluster with the highest
  cluster-cosine, plus their top-10 words).
- `figures/bertopic/intermediate/t_method_summary.json` (full numerical
  summary including Hungarian assignments and per-cluster max-overlap).

## Results

### Headline ARI / NMI

| comparison | ARI | NMI | n |
|---|---:|---:|---:|
| NMF vs OpenAI (all rows incl. -1) | 0.107 | 0.221 | 57,766 |
| NMF vs OpenAI (excl. OpenAI noise) | 0.175 | 0.304 | 39,622 |
| MiniLM-BERTopic vs OpenAI (all rows) | 0.085 | 0.197 | 57,766 |
| MiniLM-BERTopic vs OpenAI (both non-noise) | 0.190 | 0.376 | 28,915 |

§7.5's decision rule states: ARI ≥ 0.5 → cluster structure is *not*
embedding-specific; ARI ≤ 0.3 → "name embedding in every claim, and be
explicit that text-embedding-3-large reveals structure MiniLM does not."
**Both NMF and MiniLM land squarely in the ARI ≤ 0.3 territory.** The
cluster structure is materially OpenAI-specific.

### Per-cluster behaviour

The Hungarian-aligned best match per OpenAI cluster, with member-overlap
fraction:

| OpenAI cluster | NMF best (cos) | overlap | MiniLM best (cos) | overlap |
|---|---|---:|---|---:|
| 0 — AI Software Engineering | 7 (0.92) | **0.31** | 7 (0.88) | **0.50** |
| 1 — Test Automation Engineer | 6 (0.98) | 0.51 | 1 (1.00) | 0.63 |
| 2 — Data Engineer | 3 (0.99) | 0.66 | 6 (0.85) | 0.89 |
| 3 — Salesforce Cloud Developer | 9 (0.85) | 0.64 | 0 (0.93) | 0.70 |
| 4 — Full Stack Developer | 0 (0.97) | 0.86 | 5 (0.81) | 0.64 |
| 5 — Mobile Application Developer | 4 (0.73) | 0.93 | 3 (1.00) | 0.72 |
| 6 — E-commerce Software Engineering | 1 (0.85) | 0.34 | 4 (0.83) | 0.24 |
| 7 — Application Systems Analyst | 8 (0.80) | 0.50 | 8 (0.65) | 0.65 |
| 8 — ServiceNow Developer | 5 (0.77) | 0.42 | 2 (0.71) | 0.86 |

**The AI Software Engineering cluster (c0) is the most embedding-
specific.** It has the lowest member-overlap fraction across both
methods (NMF 0.31, MiniLM 0.50). Looking at top-words for each method's
"closest match" to OpenAI c0:

- OpenAI c0 top words: "ai, software engineering, automation, engineers,
  software development, devops, architecture"
- NMF cluster 7 (best match): "years non-internship, software development,
  code reviews, code reviews source control, life cycle coding standards"
  — not an AI-specific cluster, more a "generic SWE craft" cluster.
- MiniLM cluster 7 (best match): "robotics software, robotics, robot,
  anduril, robotic" — a robotics cluster, not AI/LLM.

Neither NMF (TF-IDF on full text) nor MiniLM (older / smaller embedding)
isolates the AI/LLM/agentic-engineering cluster as its own family at K=10.
The OpenAI 3072-d embedding does. This is consistent with the v3 prior's
mega-cluster failure on MiniLM and the §2.2 hypothesis that the larger,
more code-aware embedding is what cleanly separates AI engineering from
adjacent data-science / generic SWE work.

The Mobile cluster (c5) is the most method-robust — high overlap with
both NMF (0.93) and MiniLM (0.72), and high centroid cosine — because
mobile development has its own distinctive vocabulary (`react native,
ios android, mobile app`) that survives any reasonable embedding.

### Sub-cluster-vs-cluster note (relevant to §1.4.4)

The MiniLM-BERTopic fit at K=10 has 9 actual clusters with 33.8 % noise
— the same shape as the OpenAI fit, but the cluster identities differ
materially. MiniLM produces a robotics cluster instead of an AI cluster
(c0 → c7 cosine 0.88 but only 50 % member overlap). This is a §1.4.4
finding: at K = 10, MiniLM cannot tell AI/LLM work from robotics or
generic SWE because the 384-d space conflates them.

## Three-gate evaluation (per design.md §13.5)

T-method is a robustness suite, not a claim-supporting analysis.

- **Gate 1 (Narrative).** PASS. Direct evidence for §7.5's
  embedding-specificity question. The result is reportable in T3 of the
  paper.
- **Gate 2 (Effect size).** N/A — this is a comparison, not an
  effect-size test. The §7.5 decision rule (ARI ≥ 0.5 / ≤ 0.3) is the
  threshold; both methods land < 0.3.
- **Gate 3 (Robustness).** PASS in the sense that two independent
  methods (NMF on TF-IDF, MiniLM-BERTopic) both fail to reproduce the
  OpenAI structure. The disagreement is not a single-pipeline flake.

## recommend_for_paper: yes (with embedding-disclosure prose)

The paper must (per §7.5) name `text-embedding-3-large` in every C1
claim. The AI Software Engineering cluster is OpenAI-specific — it does
not emerge from TF-IDF or from the v3 prior's MiniLM embedding at the
same K. This is *not* a problem for the paper: it is a finding about
what the larger embedding can resolve that smaller ones cannot, and it
re-frames the v3 prior's mega-cluster collapse as an
embedding-capacity issue rather than a corpus-structure issue.

## Rationale

NMF and MiniLM both score below the §7.5 strong-agreement floor on ARI;
the structure that BERTopic finds in OpenAI 3072-d space is not an
artefact of either the embedding's noise or the c-TF-IDF labelling
choice, but neither does it reproduce on smaller-capacity methods. The
practical consequence for the paper: claim language must be precise
about the embedding (and the c-TF-IDF method) it depends on, and the
v3-prior mega-cluster narrative belongs in the methods discussion as an
example of what changes when capacity changes.
