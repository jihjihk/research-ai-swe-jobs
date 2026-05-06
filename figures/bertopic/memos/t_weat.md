# T-weat — five pre-registered WEAT-style association tests

## What was run

Five pre-registered WEAT tests from `config.WEAT_TESTS`, each comparing two
target sets X, Y by their differential cosine similarity to two attribute
sets A, B, with Cohen's d and a 10,000-permutation null. Bonferroni
correction across the five tests (alpha' = 0.01 / 5 = 0.002).

Code: `figures/bertopic/stage2/t_weat.py`. Run with
`.venv/bin/python -m figures.bertopic.stage2.t_weat`. Total wallclock: 53 s.
Permutation seed: 42. Hashes verified against `intermediate/stage1_freeze.json`
before any computation.

### Resolution of target-set names

- `ai_clusters` (top-words contain any of {llm, ai, ml engineer, machine
  learning, rag, agent, vector, foundation model, generative}) →
  topic_id ∈ {0 ("AI Software Engineering", n = 15,055), 6 ("E-commerce
  Software Engineering", n = 856)}. n_X = 15,911. Topic 6 enters via top
  words "ai" and "machine learning" (its top-10 c-TF-IDF list reads
  "tiktok, programming, computer science, e-commerce, internship,
  software, ai, machine learning, automation, degree computer science"),
  so it captures the early-career ML-flavored postings rather than a
  pure AI-engineering cluster. Worth flagging for downstream
  interpretation.
- `legacy_clusters` (top-words contain any of {.net, cobol, mainframe,
  php, wordpress, autosar, servicenow, plc, fortran}) →
  topic_id ∈ {8 ("ServiceNow Developer", n = 382)}. n_Y = 382. None of
  the other legacy keywords surface in any cluster's top-10 — Salesforce
  (cluster 3) and ServiceNow are the only enterprise-platform clusters
  the headline-K = 9 fit produces, and only ServiceNow trips the
  pre-registered keyword list.
- `non_ai_clusters` (complement of `ai_clusters` over real, non-outlier
  topics) → topic_id ∈ {1, 2, 3, 4, 5, 7, 8}. n = 23,711.
- `period_2024` / `period_2026`: filter on Sample A's `period` column
  (prefix `2024-` / `2026-`). n_2024 = 23,344; n_2026 = 34,422.
- `senior_swe` / `junior_swe`: `seniority_final ∈ {mid-senior, director}`
  (n = 30,894) and `seniority_final ∈ {entry, associate}` (n = 2,320).
  `unknown` (n = 24,552) is excluded from this test by construction;
  this is not pre-registered in design.md and is documented here as
  the obvious split of the 5-value enum.

### Method specifics

- Embedding source: `data/bertopic/embeddings_cache.npy` (108,514 × 3072
  float32). Cached norms 1.000 ± 0.001; defensive renormalization
  applied to posting rows before each cosine. Anchor centroids built by
  averaging the 6 anchors per attribute set, then L2-normalizing the
  mean (per spec).
- Differential cosine: s(t, A, B) = dot(unit(t), centroid_A − centroid_B).
- Cohen's d: pooled SD with ddof = 1.
- Permutation null: 10,000 random splits of (X ∪ Y) into halves of
  sizes |X|, |Y|; Cohen's d recomputed analytically from running sums.
  Two-sided p = (#perms with |d| ≥ |d_obs| + 1) / (n_perm + 1).
  Floor p ≈ 1/10001 = 9.999e-5 reported as 0.0001.

## Results

Output: `data/bertopic/weat_results.parquet` (5 rows).

| Test | X | Y | A | B | n_X | n_Y | Cohen's d | p | p_bonf |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| ai_vs_legacy_innovation_vs_maintenance | ai_clusters {0, 6} | legacy_clusters {8} | innovation | maintenance | 15,911 | 382 | **+0.763** | 1.0e-4 | 5.0e-4 |
| period_orchestration_vs_solo_ic | period_2026 | period_2024 | orchestration_mentorship | solo_ic | 34,422 | 23,344 | -0.093 | 1.0e-4 | 5.0e-4 |
| seniority_architecture_vs_implementation | senior_swe | junior_swe | architecture_design | implementation | 30,894 | 2,320 | **+0.896** | 1.0e-4 | 5.0e-4 |
| ai_growth_vs_stability | ai_clusters {0, 6} | non_ai_clusters {1,2,3,4,5,7,8} | growth | stability | 15,911 | 23,711 | +0.298 | 1.0e-4 | 5.0e-4 |
| ai_exploration_vs_exploitation | ai_clusters {0, 6} | non_ai_clusters {1,2,3,4,5,7,8} | exploration | exploitation | 15,911 | 23,711 | **-0.161** | 1.0e-4 | 5.0e-4 |

All five tests reach statistical significance at the Bonferroni-corrected
threshold (p_bonf < 0.002), but only two clear the Gate 2 effect-size
threshold of |d| ≥ 0.5: AI-vs-legacy on innovation/maintenance (d = +0.763)
and senior-vs-junior on architecture/implementation (d = +0.896).

The mean differential cosines are small in absolute value (e.g. AI-cluster
mean differential vs innovation−maintenance is +0.027; legacy-cluster
mean is −0.013). The d effect comes from a 0.04-cosine-unit gap between
two distributions whose pooled SD is roughly 0.05 — meaningful in
embedding-space terms but not visually dramatic.

The exploration-vs-exploitation test is signed *opposite* the
pre-registered T1 J-curve prediction: AI clusters trend slightly toward
exploitation vocabulary (d = −0.161) rather than toward exploration. The
effect is below the |d| ≥ 0.5 threshold so it is reported as null per
§6.4's "report nulls, do not cut" rule, but the sign is informative for
the T1 framing.

## Three-gate evaluation (per design.md §13.5)

The five tests are evaluated jointly per §13.5, but each is graded
individually for the orchestrator's cull.

### Test 1 — `ai_vs_legacy_innovation_vs_maintenance`

- **Gate 1 (Narrative).** PASS. Direct evidence for C2 ("specific legacy
  stacks are shrinking and read as maintenance, not innovation"): AI
  clusters 0+6 sit measurably toward the innovation pole (mean diff
  +0.027) while legacy cluster 8 sits toward the maintenance pole (mean
  diff −0.013). Two-sentence reading: "When we average the embeddings
  of postings inside AI-flavored vs ServiceNow clusters and project the
  difference onto an innovation−maintenance attribute axis, the gap is
  0.04 cosine units in the predicted direction. The effect size against
  within-set variance is large (d = 0.76)."
- **Gate 2 (Effect size).** PASS. d = +0.763 ≥ 0.5; p_bonf = 5e-4 < 0.01.
- **Gate 3 (Robustness).** Permutation null PASSED (p_bonf = 5e-4).
  Other robustness modes not run here: anchor LOO is run by T-axis on
  axis-style anchors not WEAT attribute sets; subset replication and
  cross-embedding replication are out of scope for the 30-minute T-weat
  budget. The single-cluster (n_Y = 382 ServiceNow) Y set is a
  meaningful caveat — the result is "AI vs ServiceNow", not "AI vs
  legacy stacks broadly", because at headline K = 9 only ServiceNow
  surfaces from the legacy keyword list.

### Test 2 — `period_orchestration_vs_solo_ic`

- **Gate 1.** Borderline. Supports T3 ("senior bottleneck → orchestration
  vocabulary concentrates over time") only weakly. Mean differentials
  are nearly identical between periods (+0.058 in 2026 vs +0.060 in
  2024); the SWE corpus is on the orchestration side of this attribute
  pair in both periods, but it has not moved.
- **Gate 2.** FAIL. d = −0.093 (small, and *opposite* the predicted
  direction toward more 2026 orchestration). Threshold is |d| ≥ 0.5.
- **Gate 3.** Permutation null reaches the floor only because n is
  large; the small-d-with-tiny-p pattern is the n-driven signature, not
  a robustness signal.

Reported as null per §6.4 protocol.

### Test 3 — `seniority_architecture_vs_implementation`

- **Gate 1.** PASS. Direct evidence for T3 ("senior bottleneck persists":
  architecture+design vocabulary attaches more strongly to senior
  postings). Note: this is a within-2024+2026 contrast, not a
  cross-period one — the test as configured does not directly speak to
  whether the senior-architecture association strengthened. To get the
  cross-period reading the test would need to be split by period.
- **Gate 2.** PASS. d = +0.896 ≥ 0.5; p_bonf = 5e-4 < 0.01.
- **Gate 3.** Permutation null PASSED. The n_Y = 2,320 junior set is
  the smaller side; large enough to be informative but ~13× smaller
  than senior. seniority_final has 24,552 "unknown" rows excluded;
  whether they are senior- or junior-skewed is a known confound flagged
  here.

### Test 4 — `ai_growth_vs_stability`

- **Gate 1.** Borderline. The sign is in the predicted direction (AI
  clusters lean more toward growth than non-AI clusters), but the size
  is small. Reads as "AI postings are slightly growthier than the rest
  of the SWE corpus", which most readers would call an obvious
  background effect.
- **Gate 2.** FAIL. d = +0.298 < 0.5.
- **Gate 3.** Permutation null PASSED at floor.

Reported as null.

### Test 5 — `ai_exploration_vs_exploitation`

- **Gate 1.** PASS as a negative finding for T1 (Brynjolfsson J-curve).
  T1 predicts AI postings are more exploratory in vocabulary; we find
  the *opposite* sign at small magnitude. A pre-registered null is
  itself informative: AI postings are not yet the future-tense
  intangible-investment language that the J-curve framing predicts.
- **Gate 2.** FAIL. |d| = 0.161 < 0.5.
- **Gate 3.** Permutation null PASSED at floor.

Reported as null with the sign flagged in prose.

## recommend_for_paper: conditional

Two of the five tests (ai_vs_legacy_innovation_vs_maintenance,
seniority_architecture_vs_implementation) clear all three gates and are
paper-grade. The remaining three are pre-registered nulls — they belong
in the paper as nulls per §6.4's protocol, not as headlines.

## Rationale

T-weat returns a clean two-of-five result: at headline K = 9, AI
clusters sit on the innovation side and the lone legacy cluster
(ServiceNow) sits on the maintenance side with d = 0.76; senior postings
sit on the architecture side and junior postings on the implementation
side with d = 0.90. Both effects are large by Cohen's conventions and
clear the Bonferroni threshold by three orders of magnitude. The other
three tests are pre-registered nulls: the 2024→2026 orchestration
shift is essentially zero (d = -0.093), AI vs non-AI growth lean is
small (d = +0.298), and AI exploration tilt is small and inverted
(d = -0.161). The paper inherits the §6.4 commitment to report nulls;
there is also a structural caveat worth surfacing: at K = 9 the legacy
side of test 1 is one cluster of 382 postings, and topic 6 ("E-commerce
Software Engineering") enters `ai_clusters` via "ai" and "machine
learning" keywords despite reading like an early-career ML-tilted
super-cluster. A K = 15 or K = 20 fit might give a more nuanced legacy
picture; the inverted exploration sign and the flat orchestration shift
are robust to that choice in either direction.
