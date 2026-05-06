# Stage 1 freeze memo — template (filled at S1.7)

Per design.md §13.3 the orchestrator produces this memo with all nine items
answered. With autonomous-run authorization (memory:
`project_bertopic_run.md`), the orchestrator self-signs this memo with
rationale recorded in `prereg_log.md` rather than blocking on real-time
human approval.

This is the template; the filled version replaces the angle-bracketed
placeholders at S1.7 and is committed as `stage1_freeze.md`.

---

# Stage 1 freeze — <DATE>

## 1. Headline K and the §4.4 sweep

| K | n clusters | noise rate | seed-pair ARI mean | per-period centroid alignment | author interp rating |
|---|---|---|---|---|---|
| 10 | … | … | … | … | (deferred) |
| 15 | … | … | … | … | (deferred) |
| 20 | … | … | … | … | (deferred) |
| 25 | … | … | … | … | (deferred) |
| 30 | … | … | … | … | (deferred) |
| 40 | … | … | … | … | (deferred) |
| 50 | … | … | … | … | (deferred) |
| 75 | … | … | … | … | (deferred) |

Author interpretability rating is deferred per the autonomous-run
authorization (no human in the loop overnight). The other three §4.4
criteria (seed ARI ≥ 0.4, period alignment ≥ 0.85, outlier ≤ 0.40)
constrain headline K well enough; the labelling protocol (§5.2) remains
the formal point at which interpretability gets re-evaluated.

**Headline K = <HEADLINE_K>**, chosen as the smallest K satisfying the
three computable §4.4 criteria.
**Super-family K = <SUPER_FAMILY_K>**, the smaller of {10, 15} that
clears seed-pair ARI ≥ 0.4.

## 2. Headline `min_cluster_size` and §4.6 sweep

| mcs | n clusters raw | noise rate raw | adj. ARI K=30 |
|---|---|---|---|
| 10 | … | … | … |
| 20 | … | … | … |
| 30 | … | … | … |
| 50 | … | … | … |
| 70 | … | … | … |

**Headline mcs = <HEADLINE_MCS>**, the largest value satisfying §4.6
criteria (post-reduction adjacent ARI ≥ 0.7 AND noise rate in
[0.15, 0.35]). If no value satisfied both, the fallback rule chose the
mcs with the noise rate closest to the band centre; that fallback is
documented in `prereg_log.md`.

## 3. Noise rate

Raw HDBSCAN noise rate at headline (mcs, seed=42): **<NOISE_RATE> %**
before any `reduce_outliers`. Reported alongside the post-reduction
rate in T-quality (§7.10).

## 4. Seed-pair ARI at headline K

| pair | ARI | centroid alignment |
|---|---|---|
| 42 vs 1337 | … | … |
| 42 vs 2026 | … | … |
| 1337 vs 2026 | … | … |

Gate (§7.1): if any pair has ARI < 0.4 AND alignment < 0.85, fall back
to super-family granularity per §10.5. **Outcome: <SEED_GATE>.**

## 5. Mega-cluster check

Largest cluster's posting share at headline K: **<LARGEST_SHARE> %**
(cluster id <LARGEST_ID>, c-TF-IDF top words: <TOP_WORDS>).

Gate (§10.1, §13.2 S1.4): > 30 % triggers hierarchical sub-clustering
attempt. **Outcome: <MEGA_GATE>.**

## 6. AI-region structure

At least one AI-flavoured cluster present? **<AI_PRESENT>** Sub-
structure observation (§1.4.4): <AI_SUBSTRUCTURE>. Differentiated
RAG / agents / evals / foundation-model splits at headline K?
<DIFFERENTIATED>.

## 7. Determinism

Double-run at (headline mcs, seed=42) produced byte-identical labels:
**<DETERM_IDENTICAL>**. ARI between the two runs: <DETERM_ARI>. Source
of any non-identity: <DETERM_DIAGNOSIS>.

## 8. Flags for Stage 2

<STAGE2_FLAGS>

## 9. Stage 1 cluster catalog (headline K = <HEADLINE_K>)

Per cluster: c-TF-IDF top-10, KeyBERTInspired top-5, posting count,
period split (2024 / 2026), top 3 firms, top 3 metros, gpt-5.5
proposed label.

<CLUSTER_CATALOG>

## Self-sign

Per the 2026-05-06 user authorization (memory:
`project_bertopic_run.md`), the orchestrator self-signs this memo and
proceeds to Stage 2. Sign-off rationale: the seed gate, mega-cluster
gate, and determinism check all <PASS_OR_NOT>; per-period centroid
alignment at headline K is <ALIGNMENT> (≥ 0.85 required for cross-
period claims); the cluster catalog is sanity-coherent on c-TF-IDF
top-words. If any of the above flags moved to "fail" we would have
fallen back to super-family granularity rather than launched Stage 2.
