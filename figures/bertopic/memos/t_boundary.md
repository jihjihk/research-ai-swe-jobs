# T-boundary — cluster-difference vectors and boundary postings, 2024 vs 2026

## What was run

Pre-registered §6.2 analysis on the headline-K = 10 BERTopic catalog (9 non-noise
clusters; cluster ids 0–8 from `data/bertopic/topic_info.parquet`).

**Hash verification.** All five Stage 1 hashes (`model_hash`, `sample_hash`,
`embeddings_cache_hash`, `assignments_hash`, `config_hash`) verified against
`figures/bertopic/intermediate/stage1_freeze.json` before any analysis ran. The
`model_hash` is computed over the directory contents at
`figures/bertopic/intermediate/raw_fit.bertopic` (the frozen
`config.RAW_FIT_PATH`); `data/bertopic/model.bertopic` is not on disk and was
not referenced. The verification block is the first executable step in the
script.

**Cluster-pair selection.** Eight pairs covering the spec-required subset
plus adjacent neighbors. The catalog has no DevOps or SRE cluster at
headline K = 10, so the "DevOps vs SRE" pair is replaced by
Salesforce vs ServiceNow (the closest platform-developer pair); this
substitution is documented in the script and noted here.

| Pair name | A id, label | B id, label |
|---|---|---|
| AI_vs_FullStack | 0, AI Software Engineering | 4, Full Stack Developer |
| AI_vs_Data | 0, AI Software Engineering | 2, Data Engineer |
| AI_vs_Ecommerce | 0, AI Software Engineering | 6, E-commerce Software Engineering |
| Test_vs_FullStack | 1, Test Automation Engineer | 4, Full Stack Developer |
| Salesforce_vs_ServiceNow | 3, Salesforce Cloud Developer | 8, ServiceNow Developer |
| Data_vs_Ecommerce | 2, Data Engineer | 6, E-commerce Software Engineering |
| FullStack_vs_Mobile | 4, Full Stack Developer | 5, Mobile Application Developer |
| AppAnalyst_vs_ServiceNow | 7, Application Systems Analyst | 8, ServiceNow Developer |

**Method.** For each pair (A, B):
1. Drop noise postings (`is_outlier == True`).
2. Compute centroid `v_A`, `v_B` as the unweighted mean of 3072-d posting
   embeddings (OpenAI `text-embedding-3-large`, raw, not L2-normalized first).
3. Take `δ_AB = v_A − v_B`, L2-normalize.
4. L2-normalize each posting embedding (so the inner product onto `δ_AB`
   reads as a cosine projection in [-1, 1]).
5. Boundary posting: `|projection| < 0.05` per design.md §6.2.
6. Boundary fraction per period (period collapsed to year: `2024-01` and
   `2024-04` → `2024`; `2026-03` and `2026-04` → `2026`).
7. Permutation null on `Δ = frac_2026 − frac_2024`: 1,000 random permutations
   of period labels (preserving marginal counts), two-sided p as the
   fraction of null `|Δ|` ≥ observed `|Δ|`. Seed = 42.

**Substrate.** Embeddings cache only. No raw `description` text was read.
Sample A only (BERTopic was fit on Sample A). Sub-sample size after the
A∪B-cluster filter and outlier drop ranges from 989 (AppAnalyst vs ServiceNow)
to 21,837 (AI vs Data).

**Code.** `figures/bertopic/stage2/t_boundary.py`. Standalone, no shared
utility module. Run from repo root with the project venv.

**Outputs.**
- `data/bertopic/boundary_postings.parquet` — 83,366 rows
  (uid, topic_id, cluster_pair, projection, period). Some uids appear in
  multiple pairs (e.g., a topic-0 posting appears in three AI pairs); pair
  membership is the disambiguator.
- `data/bertopic/boundary_summary.parquet` — 8 rows
  (cluster_pair, topic_a, topic_b, n_total, n_2024, n_2026,
  boundary_frac_2024, boundary_frac_2026, delta, abs_delta, permutation_p,
  centroid_distance_cos).

**Time.** ~5 seconds end-to-end (memory-mapped embeddings; 8 pairs × 1,000
permutations).

## Results

| Pair | n_2024 | n_2026 | frac_2024 | frac_2026 | Δ (pp) | perm p | |Δ| ≥ 5pp? | centroid cos-dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AI_vs_FullStack | 5,965 | 11,897 | 14.33% | 9.88% | **−4.46** | < 0.001 | no | 0.115 |
| AI_vs_Data | 6,827 | 15,010 | 8.22% | 10.24% | +2.02 | < 0.001 | no | 0.124 |
| AI_vs_Ecommerce | 4,508 | 11,403 | 3.86% | 6.56% | +2.70 | < 0.001 | no | 0.152 |
| Test_vs_FullStack | 5,256 | 5,394 | 4.05% | 4.47% | +0.42 | 0.286 | no | 0.154 |
| Salesforce_vs_ServiceNow | 2,671 | 1,943 | 22.58% | 14.98% | **−7.60** | < 0.001 | yes | 0.215 |
| Data_vs_Ecommerce | 2,601 | 5,037 | 0.88% | 1.19% | +0.31 | 0.226 | no | 0.212 |
| FullStack_vs_Mobile | 2,013 | 1,852 | 4.87% | 3.94% | −0.93 | 0.175 | no | 0.163 |
| AppAnalyst_vs_ServiceNow | 691 | 298 | 3.04% | 4.36% | +1.32 | 0.358 | no | 0.259 |

**Headline observations.**

- Only one of eight pairs clears the 5pp absolute-change threshold:
  Salesforce_vs_ServiceNow at −7.60pp, with permutation p < 0.001.
  This pair is **sharpening** (boundary fraction shrinks), not blurring.
- The spec-required AI vs Backend-flavored pair (AI vs Full Stack) shows
  a **−4.46pp** change: directionally sharpening, not blurring, and
  below the 5pp threshold. This contradicts C3's predicted signature
  ("boundary-fraction between AI Engineer and adjacent clusters grows
  by ≥ 5pp"). Permutation p < 0.001, so the sharpening direction is
  itself robust — this is an informative falsification, not a noisy null.
- AI vs Data and AI vs E-commerce do go in the predicted direction
  (boundaries blur, +2.0 to +2.7pp), are highly significant under the
  permutation null (p < 0.001), but fall below the 5pp Gate 2 threshold.
- The five remaining pairs (Test vs FullStack, Data vs E-commerce,
  FullStack vs Mobile, AppAnalyst vs ServiceNow, and the AI-baselines
  failing the magnitude bar) cluster around |Δ| < 1.5pp with permutation
  p between 0.18 and 0.36 — they are flat.
- Centroid cosine-distance between paired clusters ranges 0.11 to 0.26.
  No pair is geometrically degenerate.

**Subset replication (50% random hold-in, three seeds).**

| Pair | full Δ | seed 42 50% Δ | seed 100 50% Δ | seed 7 50% Δ |
|---|---:|---:|---:|---:|
| AI_vs_FullStack | −0.0446 | −0.0411 | −0.0446 | −0.0374 |
| AI_vs_Data | +0.0202 | +0.0188 | +0.0214 | +0.0198 |
| AI_vs_Ecommerce | +0.0270 | +0.0253 | +0.0279 | +0.0209 |
| Salesforce_vs_ServiceNow | −0.0760 | −0.0991 | −0.0871 | −0.0729 |

Direction and magnitude are stable under random 50% subsampling across
three seeds. This is a subset-replication robustness check; the cluster
fits themselves are not refit (the spec forbids that).

**Notes on confounds.**

- Source × period are perfectly nested in Sample A (`kaggle_asaniczka` →
  2024-01; `kaggle_arshkon` → 2024-04; `scraped` → 2026-03 + 2026-04). A
  per-source within-period subset replication is therefore not possible
  here; that check belongs to T-bootstrap §7.6 and is out of this
  sub-agent's scope. The Δ statistics here entangle period change with
  any source-platform-specific phrasing change.
- The Salesforce vs ServiceNow sharpening is the largest signal in the
  table but on the smallest sub-population (n_2026 = 1,943) and on a
  pair whose 2024 baseline boundary fraction is already 22.6% — an
  unusually large basal entanglement. It may say more about the
  Salesforce/ServiceNow vendor-platform vocabulary tightening (or
  composition shift in the scraped 2026 mix) than about a substantive
  role-boundary signal. T-bootstrap and T-quality are better positioned
  to adjudicate.

## Three-gate evaluation (per design.md §13.5)

- **Gate 1 (Narrative).** Pass for the falsification reading. The
  boundary-postings analysis is named in §1.4.1 C3 as a positive
  predicted signature ("boundary-fraction between AI Engineer and
  adjacent clusters grows by ≥ 5pp"). The AI vs Full Stack pair fails
  that prediction in both magnitude (4.46 < 5) and direction
  (sharpening, not blurring); AI vs Data and AI vs E-commerce match the
  predicted direction but fall short on magnitude. Reported as a null /
  partial-falsification of C3's boundary sub-claim is narratively clean
  and would survive a hostile reviewer.
- **Gate 2 (Effect size).** Fail for the spec-required AI-vs-Backend
  pair (|Δ| = 4.46pp < 5pp threshold, even though permutation p
  < 0.001). Pass for Salesforce vs ServiceNow (|Δ| = 7.60pp,
  permutation p < 0.001) — but in the sharpening direction and on a
  pair that does not map to a named claim. The remaining six pairs do
  not clear Gate 2.
- **Gate 3 (Robustness).** Of the relevant checks for this analysis:
  - Permutation null: passed at p < 0.001 for the four largest-|Δ|
    pairs (AI_vs_FullStack, AI_vs_Data, AI_vs_Ecommerce,
    Salesforce_vs_ServiceNow); failed for the four small-|Δ| pairs
    (p > 0.17).
  - Subset replication (50% random hold-in × 3 seeds): direction and
    magnitude preserved for all four large-|Δ| pairs.
  - Seed reshuffle (clustering-side): not in scope here — the BERTopic
    fit is frozen and the spec forbids refitting.
  - Anchor LOO: not applicable (no anchor sets used).
  - Cross-embedding (MiniLM): out of scope for this sub-agent;
    T-method §7.5 covers it.
  Two of the five robustness checks are applicable; both pass for the
  large-|Δ| pairs.

## recommend_for_paper: conditional

## Rationale

The boundary-postings analysis does not deliver a Gate-2-passing
positive instance of C3's "AI/Backend boundary blurs" prediction: the
AI vs Full Stack pair sharpens by 4.46pp (sub-threshold and
wrong-direction), and the AI-vs-Data and AI-vs-E-commerce pairs blur
in the right direction but only 2–3pp. The one pair that exceeds the
5pp threshold (Salesforce vs ServiceNow, −7.60pp) sharpens, on a
small n_2026 = 1,943, on a vendor-platform pair that does not map to
a named claim. Reported as written, the analysis falsifies the
boundary sub-clause of C3 and does not by itself support any C1–C4
claim. It can serve as an informative null in the §"State of
transition" prose — pre-registered analyses that come out null are
themselves paper-visible per design.md §11.9 — but it should not
headline. Whether to surface it depends on how the orchestrator
weights the C3 sub-clause's failure against the C3 claim's other
within-cluster vocabulary-drift evidence (T-drift). All numerical
results and the script are reproducible from the artifacts shipped
with this memo.
