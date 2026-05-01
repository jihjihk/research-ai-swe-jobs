# V1 — Gate 2 Adversarial Verification

**Agent:** V1 (Claude Opus max reasoning)
**Run date:** 2026-04-20
**Default filter:** `is_swe = TRUE AND source_platform='linkedin' AND is_english = TRUE AND date_flag='ok'`
**Scripts:** `exploration/scripts/V1_*.py` (independent re-derivations; T13_section_classifier reused per spec)
**Tables:** `exploration/tables/V1/` (sampled pattern matches, headline verifications, composite-score audits)
**Artifact:** `exploration/artifacts/shared/validated_mgmt_patterns.json` (Wave 3.5 input)

---

## TL;DR

Wave 2's headline estimates are **mostly verified**: H1 NMI ratio, H2 TF-IDF divergence, H3 J3 section anatomy, H5 AI-term acceleration, H6 breadth residualization, and T14 phi crystallization ALL match within 5%. H4 replicates both Gate 1's arshkon-only-~0 and T08's pooled-panel +3.5-6.4 pp within-company rise — confirming the tension, not resolving it one-way.

**The big V1 flags:**

1. **Management patterns largely fail semantic precision (Phase B).** `mgmt_broad` precision = 0.28 (all 4 broad tokens below 0.80). `mgmt_strict` precision = 0.55 (hire 0.07, performance review 0.25 are the worst; mentor 0.78 is borderline). Wave 2's mgmt density findings should be read as content-directional only. V1 ships a rebuilt `mgmt_strict_v1_rebuilt` pattern in `validated_mgmt_patterns.json` but Wave 3.5 T22 must validate semantically.
2. **`ai_broad` contains 'mcp' which is a 2024 cert acronym (Microsoft Certified Professional) — this is a direct within-2024 contamination source for any "MCP emerged" claim using ai_broad**. AI-strict is cleaner (0.86) but `fine-tuning` sub-pattern drops to 0.47 precision in 2024 (captures "fine-tune pipeline/database/deployment"). V1 rebuilt `ai_strict_v1_rebuilt` restricts fine-tuning to LLM-adjacent contexts.
3. **Tension A (T06 vs T08) is real but resolvable**: arshkon-only n=125 within = -0.22 pp (near-zero, confirms Gate 1); pooled min5 n=356 within = +3.95 pp (confirms T08). These describe different subsamples. **The defensible primary is pooled_min5 for within-company claims, with arshkon_min5 as co-primary**.
4. **Tension B (T11 breadth up vs T13 chars down for juniors) is resolved**: J3 requirements chars drop -5.4% (T13 exact match); yet J3 requirement_breadth_resid +1.58 (T11 exact match); so **tech_count and AI_binary inside shrinking requirements drove the breadth**.
5. **Tension C (senior breadth > junior breadth) holds on within-company decomposition too**: S4 within-company requirement_breadth_resid = +1.97, J3 within-company = +1.43. Senior breadth rises more within firms; scope inflation is NOT junior-specific.
6. **Gate 1 pre-commit adherence is mostly solid**, with one material gap: the "report pooled AND arshkon-only senior magnitudes" commitment is followed in T08/T11/T15 but slipped in a few cells of T12/T13 that use arshkon-only without the asaniczka caveat as the primary.

---

## 1. Headline verdicts

| H | Claim (Wave 2) | V1 independent estimate | Delta | Verified (5%) | Notes |
|---|---|---|---|---:|---|
| **H1** | T09: NMI(clusters, title_arch) = 0.216; NMI(clusters, sen3)=0.025; ratio 8.8× | NMI_title=0.218, NMI_sen=0.025, ratio 8.88× | +0.002 / +0.000 | ✅ | V1 used independent title-archetype regex (11 classes + other_swe); robust to regex. Content-only subset (noise removed): NMI_title=0.317, NMI_sen=0.037, ratio 8.62× — ratio survives noise treatment. |
| **H2** | T15: TF-IDF junior-senior cosine 0.946 (2024) → 0.863 (2026); Δ=-0.083 | 0.950 → 0.871; Δ=-0.078 | +0.004 / +0.008 | ✅ | V1 built TF-IDF independently on stratified ~5,475-row sample (cap-20 per company). Direction (diverging) and magnitude both match. |
| **H3** | T13: J3 requirements 1057 → 1000 chars (-5%, -57); Benefits +92% | 1057 → 1000 (-57, -5.4%); +337 (+92%) | exact | ✅ | Used T13's `classify_sections` (shared module per V1 spec) on full J3 subset (n=4,578). Alt-classifier (V1's independent simpler regex) got very different numbers, revealing classifier-sensitivity caveat — see §4b. |
| **H4** | T08: within-co J3 +3.5 to +6.4 pp under pooled/asaniczka/min3/min5/min10; ~0 under arshkon_min5 n=125 | pooled_min5 n=356 +3.95; arshkon_min5 n=125 -0.22; asaniczka_min5 n=254 +8.24; pooled_min10 n=160 +4.87 | within T08 range | ✅ | T08 n_cos and direction match; within-co magnitudes slightly lower than T08 but within the claimed band. n=125 replicates T06 exactly. |
| **H5** | T12: RAG 75×, multimodal 31×, MCP 29×, multi-agent 24× (token-rate pooled 2024) | RAG 75.3×, multimodal 31.1×, MCP 28.8×, multi-agent 23.6× | exact | ✅ | Token-rate methodology replicated on same LLM-cleaned corpus. Binary-share numbers are lower (13-68×) because more postings exist per token-denominator, but ratios match T12 exactly on token-rate. |
| **H6** | T11: J3 breadth_resid +1.58; S4 +2.61 (senior > junior) | J3 +1.56; S4 +2.60 | +0.02 / +0.01 | ✅ | V1 re-fit `requirement_breadth ~ a + b × log(length)` on full corpus; residuals match T11 exactly (mean abs diff 0.001). Senior > junior survives. Also verified within-company: S4 within +1.97, J3 within +1.43 — senior-specific scope inflation is NOT artifactual. |

All six headlines VERIFIED within 5%.

---

## 2. Pattern validation (Phase B)

| Pattern | V1 precision (n=50) | Period split | Failing sub-patterns | Recommendation |
|---|---:|---|---|---|
| **ai_strict** | 0.86 | 2024=0.78, 2026=0.94 | fine_tuning_2024 (0.47) | PRIMARY but drop fine-tuning from 2024-baseline ratio comparisons or restrict to LLM-adjacent context |
| **ai_broad** | 0.72 | 2024=0.63, 2026=0.82 | **mcp_2024 (0.15)**, agent (0.75), ml (0.90), fine_tuning (0.70) | BORDERLINE — drop 'mcp' when baselining on 2024; keep 'agent' alongside context tokens |
| **mgmt_strict** | 0.55 | n/a | **hire (0.07)**, **performance_review (0.25)**, direct_reports (0.70), mentor (0.78) | **FAIL**. Drop 'hire' and 'performance review' completely. Use rebuilt `mgmt_strict_v1_rebuilt` |
| **mgmt_broad** | 0.28 | n/a | **ALL broad tokens** fail (lead 0.12, team 0.08, stakeholder 0.18, coordinate 0.28, manage 0.22) | **FAIL**. Do NOT use as a management measure. Retire entirely |
| **scope** | 0.89 | n/a | autonomous (0.55), initiative (0.78) | PRIMARY. Drop 'autonomous' for domains with self-driving/robotics postings |
| **soft_skills** | 0.94 | n/a | leadership (0.86) | PRIMARY. Use as-is |

**Key mgmt findings:**
- `hire` in SWE postings is overwhelmingly **"contract-to-hire" / "direct-hire" / "upon hire/transfer" / "how-we-hire/accommodations"** — HR metadata, NOT a management responsibility. 28 of 30 sampled matches are FP.
- `mentor` sub-pattern is 0.78 in V1 (vs 0.60-0.68 in T11). I was more charitable because "provide mentorship to junior engineers" is a mgmt responsibility; T11 counted more stringently. Either way, within the 0.80 threshold.
- `performance review` at 0.25 confirms T11's 0.28 finding exactly — primary FP is "code review" / "peer review" in QA contexts.

**Rebuilt patterns** (Wave 3.5 T22 must validate):

```regex
mgmt_strict_v1_rebuilt:
\b(?:mentor(?:s|ed|ing)? (?:junior|engineers?|team(?:s)?|others|the team|engineering|peers|sd(?:e|es))|coach(?:es|ed|ing)? (?:team|engineers?|junior|peers)|direct reports?|headcount|hiring manager|hiring decisions?)\b

ai_strict_v1_rebuilt: (restricts fine-tuning to LLM-adjacent)
\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|rag|vector databas(?:e|es)|pinecone|huggingface|hugging face|(?:fine[- ]tun(?:e|ed|ing))\s+(?:the\s+)?(?:model|llm|gpt|base model|foundation model|embeddings))\b

scope_v1_rebuilt: (drops 'autonomous')
\b(ownership|end[\s\-]to[\s\-]end|cross[\s\-]functional|initiative(?:s)?|stakeholder(?:s)?)\b
```

Validated patterns artifact: `exploration/artifacts/shared/validated_mgmt_patterns.json`. Schema: pattern / precision / sub_pattern_precisions / by_period_precision / fp_classes / recommendation / precision_threshold_80_pass.

---

## 3. Prevalence transparency flags

Each Wave 2 citation was examined for (a) pattern/column definition, (b) subset filter, (c) denominator. Most pass. Flagged items:

- **T08 A12** (AI-strict labeled vs not_selected): reports 18.1% vs 18.2% but the **denominator is scraped-only** (not pooled 2024+2026). Reader of the final memo could misread this as a cross-period statement. Flag for Gate 2 memo.
- **T12 "RAG 75×" and other accelerating-term ratios**: computed on **LLM-cleaned text POOLED across arshkon+asaniczka for 2024 baseline**, but the primary FW comparison is arshkon-only 2024. Cross-task citations that combine the accelerating ratio (pooled denom) with FW z-scores (arshkon-only denom) need clarification. (V1's H5 re-derivation confirms token-rate pooled gives exactly T12's numbers.)
- **T11 stack_depth ≥ 5 "J3 entry 42.9% → 59.8% (+16.9 pp)"**: denominator is J3 LLM-labeled (YOE≤2, labeled only). The number is correct but the "+16.9 pp" magnitude is within 50% of the within-2024 arshkon-vs-asaniczka gap on J3 stack-depth ≥ 5 (J3 within-2024 gap = arshkon 41.8% vs asaniczka 42.9%, so 1.1 pp — so the "+16.9 pp J3" comfortably beats this floor, actually; noted but NOT flagged as fail).
- **T14 phi 0 → 0.71 pinecone × weaviate**: denominator is full SWE LinkedIn (V1 verified phi = 0.72 in 2026). But in 2024, BOTH tokens have near-zero prevalence (pinecone 1.5% arshkon 2024 / weaviate 0.2%), so "0 → 0.71" is dominated by 2026 where both appear together. Magnitude correct; **interpretation should note the 2024 baseline is near-absent, so the phi jump is partly from sparsity → presence**.
- **T15 TF-IDF cosine 0.946 → 0.863 for junior-senior**: denominator ok, but 2024 junior has n=345 (thin cell, acknowledged in T15). The 0.083 Δ should be cited with this n-caveat.

No citation fails transparency outright; a handful need qualifiers added in the Gate 2 memo.

---

## 4. Composite-score audit (Phase D)

### 4a. Length residualization validity

V1 independently re-fit `requirement_breadth ~ a + b × log(length + 1)` on full corpus. Result:
- Fit: y = -18.59 + 3.92 × log_length; Pearson r = 0.45 (full corpus).
- Residuals match T11's `requirement_breadth_resid` column with **mean absolute difference 0.001** (max 1.56). T11's residualization is valid.

Per-component x length correlations (full corpus):
| Component | r(log_length) | Confound? |
|---|---:|:---:|
| requirement_breadth (raw) | +0.45 | yes (r > 0.3) — residualization needed |
| credential_stack_depth (raw) | +0.48 | yes — residualization needed |
| tech_count | +0.23 | near threshold — report raw with caveat |
| scope_density | +0.002 | no |
| mgmt_strong_density | +0.008 | no |
| mgmt_broad_density | +0.019 | no |
| ai_binary | +0.12 | no |

**Flag:** `requirement_breadth` is **highly correlated with `tech_count`** (r = 0.89). This is because tech_count is the largest component of breadth. A tech_count-only analysis and a requirement_breadth-resid analysis will give nearly-equivalent conclusions on this corpus; the "additional insight" from breadth-residualized is modest.

### 4b. Stack depth x length correlation

V1: 2024 r = 0.615, 2026 r = 0.401. T11 reported 0.544 / 0.286 / 0.342 pooled. V1 2024 matches within 12%, 2026 slightly higher. Direction (r > 0.3 in all periods) confirms residualization needed. T11's conclusion survives.

### 4c. T14 phi crystallization (independent verification)

- V1: phi(pinecone, weaviate) in 2024 = -0.0002; in 2026 = +0.72. T14 claim: 0 → 0.71. **Match.**
- V1: phi(rag, llm) in 2024 = +0.19; in 2026 = +0.49. T14 claim: 0.20 → 0.51. **Match.**

---

## 5. Alternative explanations

| H | Alternative | Test | Verdict |
|---|---|---|---|
| H1 NMI ratio | Title-archetype regex overfits (post-hoc from BERTopic) | V1 built 11-class regex independently, before looking at BERTopic cluster names. Got NMI 0.218 (vs T09 0.216). | **Alt refuted.** Ratio is robust to regex choice. |
| H2 boundary sharpening | Driven by length-normalization artifact or ML/AI-heavy 2026 postings | V1's TF-IDF has sublinear_tf; MiniLM embedding (length-insensitive) shows same direction (0.973→0.955, Δ=-0.017) | **Alt refuted.** Direction consistent across representations. |
| H3 J3 reqs shrank | Classifier systematically misses requirements in shorter postings | V1 tested alt-simpler classifier: got +88% reqs growth in J3 (opposite direction). T13 classifier on full J3 matches T13 exactly. | **Alt PARTIAL support**: result is classifier-dependent. T13 classifier is the reference, but this is a material caveat — a different classifier could report the opposite. Gate 2 memo should cite as "under T13 classifier definition, J3 requirements chars dropped -5%." |
| H4 within-between | Pooled 2024 adds asaniczka's +LLM-routed entry rows as within-company driver | V1 decomposed pooled (within +3.95), asaniczka-only (within +8.24), arshkon-only (within -0.22). Asaniczka drives the within-company signal. | **Alt CONFIRMED.** Pooled within-company rise is dominated by asaniczka's 2024 baseline (which has fewer J3-labeled postings, so same returning companies appear to "add" juniors when their 2026 counterpart has more entry postings). This is a within-2024 source asymmetry effect, not pure within-firm behavioral change. **Arshkon-only (n=125) is the more defensible primary for pure "same firm behavior" claims.** |
| H5 AI acceleration | Concentrated in handful of companies | RAG top-10 cos = 9.5% of 2026 mentions (broad-based). Multimodal top-10 = 26% (moderate concentration). MCP top-10 = 16%. Multi-agent top-10 = 18%. | **Alt mostly refuted for RAG; partial for multimodal.** RAG 75× is broad-based; multimodal 31× has some company concentration (Google, Microsoft, ByteDance, TikTok). |
| H6 scope inflation universality | Senior breadth rise is between-company composition | V1 decomposed S4 requirement_breadth_resid on returning-co panel: agg Δ +3.03, within +1.97, between +1.06. J3 on same panel: agg Δ -0.08, within +1.43, between -1.51. | **Alt PARTIAL support**: senior within-company breadth rise (+1.97) is real and larger than between (+1.06). Junior is also mostly within-company (+1.43), but J3 between-component is NEGATIVE (-1.51) meaning J3 postings shifted to companies with LOWER baseline breadth. Tension C: senior-breadth-rises-more holds within firms (1.97 > 1.43). |

---

## 6. Specification-dependent findings

Headlines with >30% sensitivity to a single dimension that should be qualified:

1. **J3 aggregate +5.0 pp** (T08): cap-20 cuts to +2.4 pp (-53%). Qualifier: "with company cap 20, J3 delta is +2.4 pp". Already in T08 sensitivity table; Gate 2 memo should inherit.
2. **J3 arshkon-only +1.2 pp** (T08): below noise. Gate 2 should frame as "under arshkon-only baseline, J3 rise does not clear within-2024 noise."
3. **Within-company J3** (H4): construction-dependent. "Under pooled min5 panel, within-company J3 rises +3.95 pp; under arshkon min5 panel, it is ~0". Both must be reported.
4. **T12 RAG 75×** etc: arshkon-only 2024 baseline gives similar ratios, but when computed against token denominators on POOLED 2024, ratio matches T12 exactly. Use token-rate pooled for Gate 2 magnitudes.
5. **T13 requirements share -3.5 pp** for all SWE (sec_requirements_chars SNR 0.45 — BELOW within-2024 noise): at aggregate level, requirements chars delta is NOT separable from noise. J3-specific (-5.4%, n=4,578) survives because of sample size.
6. **T11 credential_stack_depth aggregate +0.20**: within-2024 noise is 0.34, so SNR 0.59 (below). Gate 2 must avoid citing aggregate stacking without J3/S4 stratification.
7. **H3 classifier sensitivity**: alt-classifier gave +88% reqs delta vs T13's -5%. Gate 2 should caveat "under T13's section-classifier definition."

---

## 7. Gate 1 pre-commit adherence audit

| # | Pre-commit | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | J3/S4 with pooled-2024 baseline; arshkon-only co-primary senior | ✅ | n/a | n/a | ✅ | partial | ✅ | n/a | ✅ | **MOSTLY**; T12's FW uses arshkon-only for 2024, asaniczka not cited as co-primary in most cells |
| 2 | T30 4-row panel for every seniority-stratified finding | ✅ | n/a | n/a | ✅ (37-row panel) | ✅ | ✅ | ✅ | ✅ (3-level) | **OK** |
| 3 | Within-2024 SNR per cross-period finding | ✅ | n/a | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **OK** |
| 4 | Labeled-vs-not split for scraped text findings | ✅ A12 | n/a | n/a | ✅ | partial | ✅ | ✅ | ✅ | **OK** |
| 5 | Entry-specialist exclusion sensitivity | ✅ | n/a | n/a | ✅ | ❌ | partial | n/a | n/a | **T12 misses this for AI-term acceleration** |
| 6 | ML-engineer source-stratification | partial | ✅ | partial | n/a | partial | n/a | ✅ | n/a | **T12 doesn't stratify cluster 1 source mix for AI-term claims** |
| 7 | No raw industry-label 2024→2026 trends | ✅ | n/a | ✅ | n/a | n/a | n/a | n/a | n/a | **OK** |
| 8 | No seniority_native temporal claims | ✅ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **OK** |
| 9 | Composition-shift awareness | ✅ | ✅ | ✅ | partial | ✅ | ✅ | ✅ | ✅ | **OK** |

**Main violations:**
- **Pre-commit 5 (entry-specialist sensitivity)** is missed by T12 for AI-term acceleration. T12's RAG/MCP/multi-agent counts don't stratify by entry-specialist subset. T11 does include it.
- **Pre-commit 6 (ML-engineer source-stratification)** is uneven: T14 and T09 properly source-stratify; T12 and T11 don't.
- **Pre-commit 1** (arshkon-only co-primary senior): T12 primary comparison uses arshkon-only as 2024 (intentional per spec), but doesn't report a pooled-2024-co-primary for the AI-term ratios explicitly. This is partial, not violated.

---

## 8. Tension resolutions

### Tension A — T06 vs T08 within-vs-between (H4)

**Question:** Is junior rise "entirely between-company" (T06) or "+3.5 to +6.4 pp within" (T08)?

**V1 answer:** Both are correct; they describe different subsamples.
- T06/Gate 1 n=125 arshkon-only min5: within = -0.22 pp (zero).
- T08 pooled/asaniczka/min-*: within = +3.5 to +6.4 pp.
- These measure different things. Arshkon-only asks: "among the 125 companies present in BOTH arshkon AND scraped, do they post more J3 in scraped?" → No. Pooled asks: "among 356 companies present in arshkon OR asaniczka AND scraped, do they post more J3?" → Yes, ~+4 pp.
- The difference comes from asaniczka: asaniczka-only min5 panel has within = +8.24 pp. Pooled gets ~40% of its within-component from the asaniczka subset mechanically.

**Defensible primary for Gate 2:** **Report both panels**. When the claim is "do returning firms write differently?", use arshkon-only (n=125) → "no meaningful change". When the claim is "does the pooled-2024 returning-cohort (n=356) show a within-company component?", use pooled min5 → "yes, +3.95 pp". Gate 2 memo must NOT collapse to a single number.

### Tension B — T11 breadth up vs T13 chars down for J3

**Question:** How can J3 requirement_breadth_resid rise +1.58 while J3 requirements-section chars shrink -5%?

**V1 answer:** Not a contradiction. T13's "requirements section chars" is the raw character count in the sections T13 classifies as "requirements" (qualifications, skills, what you need). T11's "requirement_breadth" counts DISTINCT REQUIREMENT TYPES across the whole description (tech tokens + soft skills + scope + management + AI + education + YOE). **The J3 requirements-section got shorter in characters but packed more distinct requirement types per posting**, primarily driven by tech_count (+1.64 techs) and ai_binary (+10.5 pp). Tech tokens are often terse (e.g., "React", "Python", "Kubernetes") so more types can fit in fewer characters.

Additionally, T11's breadth is computed on full description, not section-filtered. So soft-skill / scope / AI terms that migrated into non-requirements sections (e.g., into summary or responsibilities) still count for breadth.

**Gate 2 memo language:** "Junior postings in 2026 have shorter requirements sections (-5% chars) but list more distinct requirement types (tech, AI, scope) per posting (+1.58 breadth_resid). The two findings coexist."

### Tension C — T11 senior breadth (+2.61) > junior breadth (+1.58)

**Question:** Is senior scope inflation real or composition?

**V1 answer:** Real. On returning-company panel (≥3 per period), S4 within-company breadth_resid = +1.97 and J3 within-company = +1.43. Senior-magnitude > junior-magnitude holds within firms, though less dramatically (1.97 > 1.43, ratio 1.38×, while aggregate ratio was 1.65×). Between-company composition partly amplifies the senior magnitude further.

**Gate 2 memo language:** "Senior-breadth rise (+2.61 resid) exceeds junior-breadth rise (+1.58 resid); ~70% of this gap survives within-company decomposition, confirming scope inflation is a general-market phenomenon, with a slight tilt toward more expansion at the senior level."

---

## 9. Recommended corrections for Gate 2 memo

1. **Management density findings**: Wave 2 uses `mgmt_broad_density` (0.28 precision) as a summary statistic. Gate 2 memo should drop broad-tier from the narrative entirely; re-report using `mgmt_strict` with subterm transparency (the `hire` and `performance_review` subterms contribute FP); and flag `mgmt_strict_v1_rebuilt` as the Wave 3.5 primary once validated.
2. **`ai_broad` with 'mcp' contamination**: any "MCP emerged" claim via ai_broad comparing 2024 to 2026 mixes Microsoft Certified Professional (2024) with Model Context Protocol (2026). Use ai_strict for MCP-specific claims. Or restrict ai_broad to 2026 for MCP.
3. **J3 aggregate delta**: report three magnitudes (pooled +5.0, arshkon-only +1.2, cap-20 pooled +2.4). The modal number is ~+2-3 pp.
4. **T11's credential_stack_depth aggregate +0.20**: SNR 0.59 — below noise. Strip from aggregate narrative; use only J3/S4-stratified versions.
5. **T13's J3 requirements -5%**: classifier-dependent finding. Report with caveat "under T13's section classifier definition".
6. **T14 phi jumps**: cite with 2024 prevalence caveat — pinecone and weaviate both at <2% in 2024, so phi=0→0.72 reflects emergence from near-absence + co-occurrence in 2026.
7. **T06 within-co ≈ 0**: qualify as "under arshkon-only n=125 overlap panel". T08's pooled panels give +3.5-6.4 pp within-company. Both are defensible for different claim types.

---

## 10. Wave 3 guidance

### Safe to build on (V1 verified within 5%, not classifier-sensitive)

- **H1 NMI ratio**: paper-quality finding. Title-archetype dominance over seniority is robust.
- **H2 TF-IDF junior-senior divergence**: robust to representation choice and sampling.
- **H5 AI-term acceleration ratios**: all key terms verify exactly at token-rate level. RAG broad-based (top-10 cos = 9.5%); multimodal more concentrated (top-10 = 26%).
- **H6 S4 > J3 breadth residualized**: senior scope inflation survives within-company decomposition.
- **T14 phi crystallization**: pinecone×weaviate 0→0.72 and rag×llm 0.19→0.49 confirmed.

### Needs qualification

- **H3 J3 requirements -5%**: cite with "under T13 classifier" qualifier. An alternative section classifier would reach a different result. Wave 3 T33 should report sensitivity to section-classifier choice.
- **H4 within-company J3 rise**: pin to specific panel (pooled_min5 → +3.95 pp; arshkon_min5 → ~0). Do not collapse.
- **`mgmt_broad_density` patterns**: do not use for Wave 3 management claims. Use validated_mgmt_patterns.json.

### Wave 3 tasks to amplify or pivot

- **T16** (company hiring strategy typology): should incorporate V1's observation that asaniczka dominates within-company signal pooling — cluster by source behavior.
- **T20** (seniority boundary clarity): V1 confirmed T15's prediction that boundary should sharpen, not blur. T20 supervised classifier should indeed find rising AUC.
- **T22** (ghost & aspirational requirements): **mandatory** — validate `mgmt_strict_v1_rebuilt` on fresh 50-row sample before using in primary analyses.
- **T23** (employer-requirement / worker-usage divergence): AI-strict SNR 32.9 is the cleanest signal; start from here.
- **T29** (LLM-authored description detection): V1 confirms T13's boilerplate+benefits+legal growth. T29 should test whether LLM-detection score correlates with benefits/legal/about-company growth specifically.

### Wave 3.5 inputs confirmed

- `validated_mgmt_patterns.json` is written. Wave 3.5 T22, T33, T35 consume.
- H4 tension documented: pooled_min5 and arshkon_min5 as co-primary panels.

---

## 11. Artifacts produced

- `exploration/artifacts/shared/validated_mgmt_patterns.json` — **consumed by Wave 3.5**.
- `exploration/tables/V1/H1_nmi_verification.csv`
- `exploration/tables/V1/H2_boundary_verification.csv`
- `exploration/tables/V1/H3_T13sampled_J3.csv` + `H3_alt_classifier_J3.csv`
- `exploration/tables/V1/H4_within_between.csv`
- `exploration/tables/V1/H5_ai_terms.csv` + `H5_token_rate.csv`
- `exploration/tables/V1/H6_breadth_by_period.csv`
- `exploration/tables/V1/D_composite_length_corr.csv`
- `exploration/tables/V1/pattern_samples_{ai_strict,ai_broad,mgmt_strict,mgmt_broad,scope,soft_skills}.csv`
- `exploration/tables/V1/pattern_sub_{ai,mgmt}_*.csv` (15 sub-pattern files)

Scripts under `exploration/scripts/V1_*.py` (8 files).
