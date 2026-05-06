# T-axis — Five pre-registered semantic axes (§6.1) projected over Sample A

## What was run

Hash bundle (`stage1_freeze.json`) verified against on-disk artifacts before
any compute (5/5 match). The `model.bertopic` key resolves to
`figures/bertopic/intermediate/raw_fit.bertopic` — the freeze recorded its
SHA under `model_hash`; no separate `data/bertopic/model.bertopic` file was
materialized at Stage 1, and Stage 2 axis work does not need the BERTopic
model object (only embeddings, anchors, and assignments).

Code: `figures/bertopic/stage2/t_axis.py`. Standalone, ~330 lines.
Inputs:

- `figures/bertopic/intermediate/sample_a.parquet` (57,766 rows, joined to
  `data/bertopic/assignments.parquet` and the embedding-cache index).
- `data/bertopic/embeddings_cache.npy` (108,514 × 3072, float32, mmap loaded).
- `figures/bertopic/config.AXIS_ANCHORS` — the five pre-registered axes from
  §11.7, six anchor sentences per pole.

Pipeline per axis:

1. Build axis vector via PC1 of the (positive_i − negative_i) difference set,
   **uncentered SVD** per Bolukbasi (2016). Centered SVD was tried first and
   produced `cos(PC1, mean_diff) ≈ 0` for every axis — centering removes the
   bias direction itself; the design doc's reference is the Bolukbasi
   construction, which does not center. Sign-aligned to the mean difference;
   L2-normalized.
2. Project every Sample A posting via `cos(w, g)`.
3. Period-mean shift = mean(2026) − mean(2024). Permutation null = 1,000
   shuffles of period labels, two-sided p.
4. Anchor leave-one-out: drop pair `i` (positive_i and negative_i together,
   to keep PC1-of-differences well-defined), refit on 5 pairs, report
   max(1 − |cos(full, refit)|) over the 6 drops.
5. Held-out anchor validation: hold out 2 random pairs, refit on 4 pairs,
   project the 4 held-out anchors. Hit = positive lands positive, negative
   lands negative. 20 repeats; rate over 80 trials per axis.
6. Random-direction permutation null: 1,000 random unit vectors in 3072-d;
   percentile = fraction of nulls whose `|2026 − 2024|` shift is smaller
   than the actual axis's.
7. Per-cluster profile: groupby `topic_id ≥ 0` from `assignments.parquet`,
   report mean / q25 / q75 per cluster.

Elapsed: 110.8 s on the repo venv. Memory peak well under 3 GB (mmap
embeddings + 5 × 57k float64 projections).

Artifacts written:

- `data/bertopic/axes.parquet` — 5 rows × 9 columns.
- `data/bertopic/axis_projections.parquet` — 5 × 57,766 = 288,830 rows.
- `data/bertopic/cluster_axis_profile.parquet` — 5 axes × 9 clusters = 45 rows.
- `figures/bertopic/intermediate/axes/{axis}.npy` — 5 axis vectors.
- `figures/bertopic/intermediate/axes/summary.json` — full numerical summary.
- `figures/output/fig_axis_projection.{pdf,png}` — F9 candidate, five small
  multiples (2024 vs 2026 density overlays per axis), via `figures.style`.

## Results

### Headline table — period shift, sensitivity, robustness

| axis | period shift (2026 − 2024) | LOO max spread | shift / LOO ratio | held-out hit rate | random-dir percentile | perm null p (period shift) |
|---|---:|---:|---:|---:|---:|---:|
| ai_native_vs_traditional | **+0.0337** | 0.307 | 0.11 | 0.72 | 1.000 | 0.000 |
| ic_vs_management | +0.0047 | 0.083 | 0.06 | 1.00 | 0.881 | 0.000 |
| builder_vs_operator | +0.0051 | 0.099 | 0.05 | 0.80 | 0.913 | 0.000 |
| concrete_vs_abstract | +0.0049 | 0.067 | 0.07 | 0.90 | 0.906 | 0.000 |
| generalist_vs_specialist | +0.0018 | 0.109 | 0.02 | 0.91 | 0.479 | 0.000 |

Period-mean breakdowns (n_2024 = 23,344; n_2026 = 34,422):

| axis | 2024 mean | 2026 mean |
|---|---:|---:|
| ai_native_vs_traditional | −0.0154 | +0.0182 |
| ic_vs_management | −0.0262 | −0.0215 |
| builder_vs_operator | −0.0011 | +0.0040 |
| concrete_vs_abstract | −0.0905 | −0.0856 |
| generalist_vs_specialist | −0.0615 | −0.0597 |

### Per-cluster profile (mean projection by topic_id)

`ai_native_vs_traditional` ranks clusters cleanly along the expected axis:

| cluster | gpt5.5 label | mean |
|---|---|---:|
| 6 | E-commerce Software Engineering* | +0.052 |
| 2 | Data Engineer | +0.046 |
| 0 | AI Software Engineering | +0.027 |
| 1 | Test Automation Engineer | −0.012 |
| 3 | Salesforce Cloud Developer | −0.024 |
| 5 | Mobile Application Developer | −0.042 |
| 4 | Full Stack Developer | −0.043 |
| 8 | ServiceNow Developer | −0.049 |
| 7 | Application Systems Analyst | −0.065 |

\* cluster 6's top words include "computer science / programming / internship";
the gpt5.5 label is likely off and this looks like an internship/early-career
mass — its position on the AI-native pole is suspect and contributes to that
axis's anchor sensitivity.

`ic_vs_management`, `builder_vs_operator`, `concrete_vs_abstract`,
`generalist_vs_specialist` cluster orderings are in
`data/bertopic/cluster_axis_profile.parquet`. Notable: cluster 7
(Application Systems Analyst) is consistently the most "operator / abstract /
specialist / management" of the nine across multiple axes — face-valid for a
maintenance-flavored cluster.

### Figure

`figures/output/fig_axis_projection.pdf`: five small multiples, 2024 vs 2026
density overlays. The `ai_native_vs_traditional` panel is the only one with
a visible 2024→2026 shift; the other four panels show near-perfectly
overlapping distributions.

## Three-gate evaluation (per design.md §13.5)

### Gate 1 — Narrative

**Pass for `ai_native_vs_traditional`.** This axis is the load-bearing
geometric instrument for **C1** (AI-native role families have crystallized)
and the IC↔management axis is named directly in **T1** and §1.4.3.
A 2026−2024 mean shift on the AI axis is the predicted L3 signature for C1
in §1.4.1.

**Conditional pass for `ic_vs_management`, `builder_vs_operator`,
`concrete_vs_abstract`, `generalist_vs_specialist`.** Each maps to an
auxiliary claim — IC↔management for §1.4.3 (junior/senior re-allocation
within shared headcount), builder/operator and concrete/abstract for T1
(intangible-investment phase, exploration framings), generalist/specialist
for the v3-prior question on AI-cohort sub-structure (§1.4.4). But the
period-shift signal on these four is below noise.

### Gate 2 — Effect size

**Threshold:** period shift ≥ 0.05 cosine units **AND** ≥ 3× LOO spread.

| axis | shift ≥ 0.05? | shift ≥ 3 × LOO? | Gate 2 |
|---|:---:|:---:|:---:|
| ai_native_vs_traditional | no (0.034) | no (ratio 0.11) | **Fail** |
| ic_vs_management | no (0.005) | no (0.06) | **Fail** |
| builder_vs_operator | no (0.005) | no (0.05) | **Fail** |
| concrete_vs_abstract | no (0.005) | no (0.07) | **Fail** |
| generalist_vs_specialist | no (0.002) | no (0.02) | **Fail** |

All five axes fail Gate 2 on both legs.

The shifts that *do* exist are statistically significant (period-permutation
p = 0 on n ≈ 58k), but the effect sizes are small relative to the axes'
own anchor sensitivity, which is the point of the §11.7 pre-registered
threshold.

### Gate 3 — Robustness

| Check | Result per axis |
|---|---|
| **Anchor LOO** (max spread) | ai_native 0.31 (above §11.9 cap of 0.10 — flag); others 0.07–0.11 |
| **Held-out anchor** (≥ 0.80 hit rate per §11.9) | ai_native 0.72 (below cap), builder/concrete/generalist/IC ≥ 0.80 |
| **Permutation null on period shift** (1,000 perm) | All axes p < 0.001 — every shift is real, just small |
| **Random-direction percentile** (effect size vs 1,000 random 3072-d directions) | ai_native 1.000, IC 0.881, builder 0.913, concrete 0.906, generalist 0.479 — `generalist_vs_specialist` is at the median of random directions, i.e. the axis is **not** discriminating better than chance |
| **Subset replication** | Not run separately; design-doc spec did not require it for this task |
| **Cross-embedding** | Out of scope for T-axis (T-method handles MiniLM) |

`ai_native_vs_traditional` survives 2/4 robustness checks tested (perm-null,
random-direction). It fails LOO and held-out — the §11.9 thresholds say
either flags the axis as unstable or cuts it.

`ic_vs_management`, `builder_vs_operator`, `concrete_vs_abstract` survive
3/4 tested. `generalist_vs_specialist` survives 2/4 — its random-direction
percentile of 0.479 is a serious problem (a random direction does this
axis's job ≥ half the time).

## recommend_for_paper: no

## Rationale

All five pre-registered axes fail Gate 2 on both legs (period shift below
0.05 cosine and below 3× anchor sensitivity). The largest shift —
`ai_native_vs_traditional` at +0.034 — is statistically significant but is
also the axis with the worst anchor stability (LOO spread 0.31, three times
the §11.9 cap of 0.10) and the worst held-out validation (0.72 vs the 0.80
threshold). Cluster ordering on this axis is face-valid (AI Software
Engineering and Data Engineer at the AI-native pole; ServiceNow and Systems
Analyst at the traditional pole), which is consistent with the §6.1 axis
genuinely capturing the AI-native vs traditional contrast at the cluster
level — but the period drift on individual postings is too small relative to
the axis's anchor wobble to support a paper-visible C1 sub-claim. The four
non-AI axes show even smaller period shifts (≤ 0.005 cosine units), and
`generalist_vs_specialist` is no more discriminating than a random
3072-d direction on the period-shift task. The cluster-axis profiles
(`cluster_axis_profile.parquet`) have value as descriptive context for §1.4.3
and §1.4.4 but do not by themselves clear any §13.5 gate. The orchestrator
may want to keep the AI-native axis as an *appendix* descriptive instrument
(it ranks clusters as expected) while reporting that the §6.1 period-shift
test came in null for the paper body.
