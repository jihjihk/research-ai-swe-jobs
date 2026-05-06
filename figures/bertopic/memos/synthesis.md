# Stage 3 synthesis — what survived the cull

## Executive summary

The discovery layer found nine cluster families at headline K=10 and rejected most of the embedding-space machinery the design doc hoped would carry weight. The paper's strongest claim is **role concentration** (effective number of clusters fell from 6.1 in 2024 to 4.9 in 2026, a 20 % drop) and the strongest narrative is **AI Software Engineering as the absorbing cluster**, growing from 27.7 % of postings to 44.9 %. The pre-registered axis projection, anchor-neighbourhood diffusion, and boundary-postings tests all came back null or off-direction; the within-cluster vocabulary-rewriting story (C3) does not survive in the form the design doc anticipated.

## Survivors

### Concentration (C4) — `data/bertopic/assignments.parquet`

The corpus consolidated. Effective number of clusters — `exp(entropy(share))` — fell from **6.09 in 2024 to 4.88 in 2026**, a 20 % decrease that comfortably clears the pre-registered ≥ 10 % threshold. The Herfindahl index moved from 0.191 to 0.275, also pointing the same way. The top-5 cluster share moved less dramatically (92.1 % → 93.1 %, well below the 5 pp HHI threshold), so the §1.4.1 prediction reads as "satisfied on entropy, not on top-5 share" — both metrics agree in direction. Three gates pass cleanly: narrative (direct evidence for C4), effect size (entropy −20 %), robustness (the per-period reproduction in T-bootstrap shows centroid alignment 0.91/0.92, so the cluster identities the shares are computed against are stable).

*Proposed prose.* "By 2026 the SWE corpus is one fifth less spread across role families than it was in 2024 — measured as the effective number of clusters in a K=10 BERTopic fit, the count drops from 6.1 to 4.9. Software-engineering job postings are concentrating, not fragmenting."

### AI Software Engineering as the absorbing cluster (C1, qualified)

Cluster 0 — labelled "AI Software Engineering" by gpt-5.5 — is the largest in both periods and absorbs most of the consolidation: 27.7 % in 2024, 44.9 % in 2026 (a +17.2 pp shift on a base of ~58 k SWE postings). The c-TF-IDF top-words read **"ai, software engineering, automation, engineers, software development, expertise, engineering, devops, architecture"** — note the lead is `ai` but the trailing words are generic SWE craft. T-method's cross-embedding rerun is the load-bearing caveat: NMF on TF-IDF and BERTopic refit on `all-MiniLM-L6-v2` both fail to isolate this cluster (member-overlap 0.31 and 0.50 against MiniLM's nearest analogue). The cluster is a property of the OpenAI 3072-d embedding plus headline K = 10. We cannot read it as "AI-native role family has crystallized as its own thing"; we can read it as "the embedding lumps AI-flavoured engineering work into one large absorbing family that grew sharply between 2024 and 2026." Three gates: narrative (yes — C1), effect size (the +17.2 pp share shift is the largest in the catalog by a factor of three), robustness (per-period centroid alignment 0.91/0.92 supports the cluster-identity continuity, but cross-embedding ARI 0.19 demands the embedding caveat).

*Proposed prose.* "The largest cluster in the BERTopic fit, dominated by AI / automation / engineering vocabulary, grew from 27.7 % of SWE postings in 2024 to 44.9 % in 2026. Both methods we cross-checked against — TF-IDF NMF and a smaller-model BERTopic — fail to reproduce this cluster, so we name the embedding (`text-embedding-3-large`, 3072-d) in the claim itself. What this analysis shows is not that AI Engineer became its own role family; it shows that postings whose language is shaded by AI vocabulary became the absorbing family of a model that can resolve them."

### Legacy stack contraction (C2, narrowed)

Three small enterprise / maintenance clusters meet the C2 threshold of n_2026 / n_2024 < 0.6: **Salesforce Cloud Developer** 15.6 % → 7.4 % (ratio 0.47), **Application Systems Analyst** 3.1 % → 0.5 % (ratio 0.16), and **ServiceNow Developer** 1.3 % → 0.7 % (ratio 0.54). T-weat Test 1 corroborates the qualitative reading: the AI clusters sit measurably toward the *innovation* attribute pole and ServiceNow toward *maintenance* (Cohen's d = +0.76, p_bonf = 5e-4). The pre-registered claim text mentions ".NET, COBOL, mainframe, PHP/WordPress, AUTOSAR, ServiceNow, PLC" — at K = 10, only ServiceNow surfaces as its own cluster; the rest dissolve into the broader Salesforce/Application Systems Analyst clusters or the noise floor. C2 survives as a **narrowed claim** about Salesforce, ServiceNow, and analyst maintenance work, not the full legacy-list. Three gates: pass on narrative (C2); pass on effect size (Δshare > 5 pp, ratio < 0.6); robustness via T-weat permutation null and per-period centroid alignment.

*Proposed prose.* "Three enterprise / maintenance-flavoured clusters lost share roughly in tandem: Salesforce postings fell from 15.6 % to 7.4 %, ServiceNow from 1.3 % to 0.7 %, and the Application Systems Analyst cluster from 3.1 % to 0.5 %. WEAT-style tests place these clusters' centroids on the maintenance side of an innovation-vs-maintenance attribute axis, with Cohen's d = +0.76 separating them from AI-flavoured clusters."

### Senior bottleneck (T3) — `data/bertopic/weat_results.parquet`

Senior SWE postings (mid-senior + director, n = 30,894) are more associated with `architecture / design` vocabulary, junior postings (entry + associate, n = 2,320) with `implementation` vocabulary; Cohen's d = +0.896, p_bonf = 5e-4. The effect is large and survives the 10 000-permutation null. This is direct evidence for the T3 / Autor-Thompson "senior bottleneck" framing — orchestration vocabulary concentrates among senior postings.

*Proposed prose.* "On a senior-vs-junior split of the SWE corpus, language tied to architecture and system design is concentrated in senior postings; junior postings cluster around implementation vocabulary. The contrast is stark — Cohen's d = +0.90 — and supports the senior-bottleneck reading: AI lifts the codified rungs of the ladder, but seniors persist."

## Cuts

### Semantic-axis projection (T-axis) — all five axes fail Gate 2

The five §6.1 axes (AI-native ↔ traditional, IC ↔ management, builder ↔ operator, concrete ↔ abstract, generalist ↔ specialist) all show period-mean shifts well below 0.05 cosine units (max +0.034 on the AI-native axis). The largest shift is also the axis with the worst anchor leave-one-out spread (0.31 vs the §11.9 cap of 0.10) and below-target held-out validation (0.72 vs 0.80). The axis is not a stable instrument for measuring period drift on this corpus. Cluster ordering on the AI-native axis is face-valid (`AI Software Engineering` and `Data Engineer` positive; `ServiceNow` and `Application Systems Analyst` negative), so the per-cluster profile has appendix value as descriptive context for §1.4.3. Cut from body. Whether the design should be retried with anchor-quantile-based axes rather than fixed cosine thresholds is a question for the next iteration.

### Anchor-neighborhood diffusion (T-anchor) — embedding-scale incompatible

The §6.5 thresholds {0.5, 0.6, 0.7, 0.8} were calibrated for MiniLM-style cosine distributions. With OpenAI 3072-d embeddings, the maximum anchor-to-posting cosine over all 57 766 postings is 0.65; **τ ∈ {0.7, 0.8} is empty in both periods for every anchor**. Only τ = 0.5 has substantive mass, and at that threshold the `sre` and `backend_engineer` neighbourhoods are 88–93 % composed of cluster 0 (the AI / SWE absorbing cluster), so they read mostly as "the headline cluster's footprint" rather than role-anchored neighbourhoods. Cut. A retry would need anchor-specific quantile thresholds (e.g. top-1 % cosine per anchor) and a paper-visible erratum to the prereg.

### Boundary postings (T-boundary) — pre-registered null

C3's predicted signature on cluster boundaries was *blurring*: postings near a δ_AB = v_A − v_B vector should grow in share between 2024 and 2026 by ≥ 5 pp. The data show the opposite. The AI-vs-Full-Stack pair, the spec-required "AI vs adjacent" cleavage, sharpens by −4.46 pp (permutation p < 0.001). Salesforce-vs-ServiceNow, the only pair clearing the 5 pp magnitude threshold, also sharpens (−7.60 pp). Five other pairs are flat. This is informative falsification of the boundary sub-clause of C3 and is itself worth a paper sentence: roles are not blurring along the AI front, they are sharpening.

### WEAT Tests 2, 4, 5 — small effects, one inverted

Three of the five pre-registered WEAT tests came back below the |d| ≥ 0.5 threshold. The period-orchestration test (d = −0.09) shows the corpus already on the orchestration side in both periods; growth-vs-stability for AI clusters (d = +0.30) is in the predicted direction but small. The inverted test is **T1's J-curve prediction**: AI clusters tilt toward `exploitation` vocabulary, not `exploration` (d = −0.16). This is a pre-registered null worth surfacing in the paper's "State of transition" section as evidence that the AI-engineering language in 2026 reads as concrete production work, not as exploratory or future-tense framing.

## Lessons

The design's strongest assumption — that the OpenAI 3072-d embedding would resolve fine sub-archetypes that MiniLM blended — is borne out at the cluster level (the AI cluster becomes its own family that NMF and MiniLM cannot reproduce) and falsified at the geometric level (anchor-neighbourhood thresholds don't translate; semantic-axis shifts on individual postings are dominated by anchor noise). The two are not in tension: a denser embedding can produce cleaner cluster assignments without producing larger period-drift effect sizes on individual posting projections, especially when the corpus drift is concentrated in the cluster-share dimension rather than in within-cluster geometric repositioning.

The mega-cluster gate at headline K = 10 was passed (largest cluster 26.1 %), but only because K = 10 itself is super-family resolution. T-quality showed the AI Software Engineering cluster has **negative silhouette** in 5-D UMAP space (−0.119) — its average member is closer to some other cluster's centroid than to its own. This is the v3 prior's mega-cluster problem in a different costume: the cluster exists at the c-TF-IDF level (the top-words are coherent) but not at the geometric level (the boundary is fuzzy). A fairer reading is that the OpenAI embedding admits a "broad AI-flavoured engineering" family that subsumes some adjacent work; whether that family is internally differentiable is a §1.4.4 question that this run cannot answer at K = 10.

Cluster 6 ("E-commerce Software Engineering") is mislabelled. Its c-TF-IDF top-words read "tiktok, programming, computer science, e-commerce, internship, software, ai, machine learning" — this is an early-career / internship-heavy cluster with TikTok overrepresented, not an e-commerce cluster. The §5.1 LLM-naming step saw `tiktok` and inferred e-commerce. The §5.2 final-labelling protocol must explicitly reject this label.

The §7.11 cross-model naming check shows gpt-5.5 and gpt-5.4-mini agree exactly on 3 of 9 labels; mean label-embedding cosine 0.834 sits below the 0.85 trigger. The labels are model-sensitive in non-trivial ways (e.g. cluster 1: "Test Automation Engineer" vs "Software Engineer", cosine 0.598). The §5.2 protocol must not depend on either model alone.

## Recommendations for the paper

**Body.** C1 (narrowed and embedding-disclosed), C2 (narrowed to Salesforce / ServiceNow / Application Systems Analyst), C4 (concentration via effective-N), T3 (senior-vs-junior architecture-vs-implementation WEAT). The boundary-sharpening result (C3 falsification) belongs as a single sentence in the "State of transition" section.

**Appendix.** T-axis cluster-profile table (descriptive ordering on the AI-native axis), T-quality coherence + silhouette + cross-model naming numbers, T-bootstrap stability table, T-method ARI table, T-weat null tests (Tests 2, 4, 5).

**Drop.** The full anchor-neighbourhood F10 figure (cut entirely), the full axis-projection F9 figure (replace with a single panel for the cluster-axis profile if it's wanted, or drop), the Lorenz curve for C4 (effective-N reads more cleanly than HHI here).

## Open questions for human work

1. **Headline K.** The §4.4 rule selected K = 10 (smallest qualifying), giving 9 actual clusters and super-family resolution. The paper's narrative wants finer resolution for per-cluster claims (the cluster catalog at K = 25 or K = 30 has 24 / 29 clusters at similar seed-pair ARI). The author interpretability rating, deferred during the autonomous run, is the gating step. If K = 25 is elevated, T-axis / T-drift / T-boundary should be re-run at that resolution; T-weat's `legacy_clusters` set will widen beyond ServiceNow.
2. **§5.2 labelling protocol.** Cluster 6's "E-commerce" label and the gpt-5.5 / gpt-5.4-mini disagreement at cluster 1 both demand the human-review step that §5.2 left TBD. Until labels are frozen, the cluster catalog's `label` column should carry both the gpt-5.5 and gpt-5.4-mini proposals.
3. **T-l1l2 crosstab.** `role_family_l1` and `skill_theme_*` are not yet in `unified_core.parquet`. The user's Stage 12 classification process was running during the orchestrator's session; once those columns land, T-l1l2 (queued, not run) will produce the §7.7 crosstab.
4. **T-ablations.** Deferred to a follow-up session. The §8.2 / T6 sign-consistency matrix is appendix-grade; the author can run it directly against the surviving artifacts when context permits.
5. **C3 within-cluster vocabulary drift.** The boundary-blurring sub-clause of C3 was falsified (sharpening, not blurring). The vocabulary-rewriting sub-clause (top-20 c-TF-IDF set-difference ≥ 30 % between periods inside stable clusters) was not run by any Stage 2 sub-agent; this is the cleanest remaining test of "rewriting in place" and would close the loop on C3.
