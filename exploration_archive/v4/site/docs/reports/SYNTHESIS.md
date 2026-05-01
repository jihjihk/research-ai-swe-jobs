# SYNTHESIS: SWE Labor Market Exploration (Waves 1-3 + V1/V2)

**Date:** 2026-04-10
**Scope:** Consolidated handoff from the exploration phase (T01-T29, V1, V2, Gates 0-3) to the analysis phase.
**Read time:** ~30 minutes. This document supersedes the chronological gate memos for any conflict; refer to them only for historical audit.

---

## 1. Executive summary

Between 2024 and 2026, the **tech-cluster occupational space** (software engineering and adjacent technical roles) restructured along three independent content axes, with one additional composition shift at the employer panel level. **First,** posting content expanded: the same companies in every tech-domain archetype began demanding more categories of requirements simultaneously (credential stack ≥7 rose 3.8% → 20.5%, a 5.4× jump; tech count mean grew ~16-27%; median tech count 3→4), postings grew longer (+88% in core sections, roughly half of which is genuine content expansion after text-source control), and AI mentions quintupled (13.7% → 52.0%). **Second,** the senior tier reorganized toward IC + mentoring + technical orchestration: explicit people-management language collapsed (-58% density), mentoring nearly doubled (11% → 22% any-share), and technical orchestration grew most (+47% density) with a new vocabulary that didn't exist in 2024 (`agentic` 415×, `prompt engineering` 20×, `guardrails` 15×, `multi-agent` 40×). The director tier merged into mid-senior on every feature except strategic scope. **Third,** entry-level posting share rose modestly (+3-6pp depending on method) but the rise is 50-87% **between-company composition** (depending on panel definition) — not within-company employer pivot. The 2026 entry pool is concentrated at a small set of formal new-grad-program employers (Google, Walmart, Qualcomm, SpaceX, Amazon, Microsoft, Cisco, Visa, and an unexpectedly over-represented defense-contractor cluster).

**What didn't change:** There is no semantic convergence between junior and senior postings (T15; T20 shows only the director↔mid-senior boundary blurred). The market's natural latent structure is tech domain, not seniority (T09 NMI: archetype × language = 0.11 vs archetype × seniority = 0.015). The restructuring is not SWE-specific — SWE-adjacent technical occupations show the same AI restructuring at ~100-105% of SWE magnitude (T18, V2), with length and scope growth actually **larger** in adjacent than in SWE proper. And employer AI requirements do **not** outpace worker AI usage: employer direct-only AI mention is ~52-56% in 2026 vs ~80% worker usage on StackOverflow, a 24-28pp gap. The original RQ3 "anticipatory restructuring" hypothesis is **inverted** — workers adopted AI first; employers are catching up at 4-10× relative rate but still below the line.

**Methodological lessons.** The exploration surfaced several measurement hazards that must be carried forward. LinkedIn's native seniority labels have differential accuracy across data snapshots (41% of arshkon `seniority_native='entry'` rows have YOE ≥ 5, vs 9.5% in scraped) — the original "junior decline" headline was a label-quality artifact. Naive keyword indicators (`hire`, bare `agent`, `mcp`) inflate findings by 3-5×. The `swe_tech_matrix.parquet` has silently broken regexes for `c_cpp` and `csharp` (`\bc\+\+\b` cannot match); C++ is actually the #3 growing systems technology, not a decliner. Cross-period length comparisons are sensitive to text-source composition (2024 is 91% LLM-cleaned vs 2026 21%). Three Wave 2/3 tasks disagreed on tech-count growth (T11 +34%, T14 +60%, T19 flat); V2 reconciled to ~16-27% mean / 3→4 median. Within-vs-between decomposition matters more than any single aggregate number: entry share is 50-87% between-company, AI mentions are 91% within-company — different mechanisms.

**What the analysis phase should test next.** (i) Re-verify every tech-matrix-dependent number under a clean 39-tech detector on raw description (V2's method). (ii) Treat the senior IC + mentoring + technical orchestration shift as the paper's sharpest narrative finding and nail its causal interpretation. (iii) Explicitly test the SWE-specific vs tech-cluster-wide framing as the default sensitivity analysis, not as an afterthought. (iv) Use T22's validated patterns for any mentoring/hedge/firm/AI measurement. (v) Apply description_hash dedup (or within-company dedup) to any entry-level analysis until the preprocessing dedup fix lands. (vi) Investigate the AI/ML heterogeneity within the senior finding (mentoring DECLINED in AI/ML while it grew everywhere else — the single most unexpected domain interaction).

---

## 2. Data quality verdict per RQ

### RQ1 — Employer-side restructuring

**Safe:**
- Scope-inflation claims (credential stack ≥7, requirement breadth, scope language, AI mention growth) under the combined best-available seniority column and under within-domain stratification. Robust to LLM authorship (T29), domain composition (T28: within-domain everywhere), text-source confound for credential stacking specifically (V1).
- Senior IC + mentoring + technical orchestration shift, using T22's validated patterns. Converges across T11, T21, T28, V1, V2.
- Within-company AI mention growth (~91% within per T16). AI restructuring is genuinely employer-level content change.

**Need caveats:**
- Length growth: absolute magnitude (+88% aggregate, +57% median core) is inflated by 2024→2026 text-source composition shift. Under LLM-text-only subset, length growth is +26% (V1). Report both: the aggregate change is real but ~half of it reflects text-source composition. The "core sections grew more than boilerplate" finding (T13) holds under text-source control.
- Tech-count growth: use **~16-27% mean / 3→4 median** (V2 independent detector). Do NOT cite T11's +34% or T14's +60%. Do NOT cite T19's "nearly flat".
- Mentoring absolute levels: T11 vs T22 patterns differ by 47-65% in absolute mention rate. Growth ratios are consistent (~2.5×). Use T22's validated patterns for any new computation; when citing T11/T21 absolute levels, note the pattern difference.

**Do not attempt:**
- Any claim about c_cpp, csharp, or possibly other specific techs that relies on `swe_tech_matrix.parquet` without cross-checking via direct LIKE queries on raw description. The matrix has known silent failures for special-character tokens and possible other bugs (see Section 4).
- Seniority-stratified claims under `seniority_native` or `seniority_final` with asaniczka pooled into 2024 (asaniczka has zero native entry labels; pooling biases the direction).

### RQ2 — Task and requirement migration

**Safe:**
- Senior-tier vocabulary migration (people-management → mentoring + technical orchestration) via T12 Fightin' Words (credential + AI subsets, which are robust under company capping per V1) plus T21's cluster analysis plus T22's validated patterns.
- New AI-orchestration vocabulary enumeration (`agentic`, `prompt engineering`, `guardrails`, `multi-agent`, `langchain`, `rag`) as a 2024→2026 emergent category. All are near-zero in 2024 and measurable fractions in 2026.
- Mid-senior credential vocabulary stripping (`qualifications`, `degree`, `bachelor`, `required`, `requirements`) — survives capping (V1 F).

**Need caveats:**
- "Downward migration" (senior responsibilities appearing in junior postings): T15 rejects semantic convergence; T20 finds only director↔mid-senior blurred (not junior↔senior). The original RQ2 directional hypothesis of senior-to-junior migration is not supported. The actual migration is **sideways within the senior tier**.

**Do not attempt:**
- Bigram-level claims from T12 uncapped (`accommodation`, `usd`, `values`, `americans`, `marrying`, `experimenting`, `empowerment`, `external`, `html`, `javascript`, `linux`, `located`, `powerful`, `qualifications`-adjacent boilerplate) that did not survive V1's capped Fightin' Words re-run. These were Amazon/volume-driven artifacts.
- Any claim that uses bare `agent` pattern. Use `agentic` (≥95% AI precision) or `AI agent` / `multi-agent system`. Bare `agent` is ~55-72% precision with legal/insurance/robotics/change-agent contamination.
- Any claim that uses `mcp` as an AI tool pattern. `MCP` matches `Microsoft Certified Professional`; it was removed from T22's validated `ai_tool` patterns.

### RQ3 — Employer AI requirement vs worker usage

**Safe:**
- Direct-only employer AI mention rate: ~14-15% (2024) → ~52-56% (2026). Growth ~4-5× (T23 and V2 match within 1pp on delta).
- StackOverflow benchmark numbers (2024 62-63% pro dev AI usage, 2025 ~80%) are within published ranges per V2 spot-check.
- The **inversion** of the original anticipatory hypothesis: employer AI requirements LAG worker AI usage by ~24-28pp; employers are catching up at 4-10× the relative worker growth rate but are still below the line. Robust across all four benchmark scenarios in T23.
- Structural AI hedge ratio: hedge:firm markers within 80-char windows of AI terms are ~10-11× in both periods (V2: 10.6→11.6; T22 reported ~10). **Temporally stable** — it's a structural grammar of how employers signal AI demand, not a 2024-2026 shift. The aggregate aspiration ratio rose as a composition effect because AI mention prevalence quintupled.

**Need caveats:**
- The absolute gap size (24-28pp) depends on benchmark pattern choice and on whether StackOverflow's selection bias (survey respondents may be more AI-adopting than the full worker pool) narrows the true gap. Report with benchmark sensitivity bands.
- The hedge:firm multiplier between AI-window and global: T22 said ~6.7×, V2 said ~4-5× (because V2's global baseline is broader). The qualitative conclusion — AI demands are substantially more hedged than non-AI demands — is robust; the multiplier magnitude is pattern-dependent.

**Do not attempt:**
- Worker-side benchmarking using anything other than StackOverflow Developer Survey and the specific SO subsets T23 used (central 80% proxy, agentic 24% weekly-plus proxy). Other benchmarks were not validated.

### RQ4 — Mechanisms

Not a data question per se — interview-driven. The exploration supplies a specific priority list for interview design (see Section 16). Qualitative instruments should focus on: (a) the mentoring growth in non-AI/ML domains vs its decline in AI/ML, (b) the director merge into mid-senior on all dimensions except strategic scope, (c) the lagged employer AI catch-up (when did your team start including AI tools in JDs?), (d) the defense-contractor over-representation in 2026 entry-level postings, (e) whether recruiters perceive the scope inflation as "we ask for everything and filter later" or as real raised hiring bars.

### Emerging RQ5 — Cross-occupation generalization

**Safe:**
- SWE-adjacent AI rate grew +30.4pp vs SWE +29.0pp (V2 broader pattern). Adjacent is at ~100-105% of SWE magnitude, not 83% as T18 originally reported under a narrower pattern.
- Length growth is larger in adjacent than SWE (+629 vs +527 chars under text-source control).
- Scope language growth is larger in adjacent than SWE (+26.3pp vs +19.0pp).
- SWE↔control CONVERGED (+0.079 cosine); SWE↔adjacent slightly DIVERGED (-0.022) — opposite of a naive "boundary blurring" prediction.
- "AI Engineer" title evolved cleanly from PyTorch/ML role (0% agentic, 40% pytorch) to LLM-agent role (45% agentic, 32% pytorch, 54% LLM mentions), with 14-16× volume growth.

**Need caveats:**
- "Embedding-adjacent"-only subset (strictest adjacent definition): AI rate grew +28.6pp on n=3,166 (2026). Finding holds even on the strictest adjacent tier.

**Do not attempt:**
- Claims that the SWE restructuring **precedes** or **follows** adjacent. The three-snapshot data structure does not support leading/lagging inference. T19 confirmed this: the data is effectively three temporal points, not a time series.

---

## 3. Recommended analytical samples

**Default filter:** `source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE`.
**Period convention:** `2024` = arshkon + asaniczka (28,232 SWE rows); `2026` = scraped LinkedIn (35,062 SWE rows). **Do not pool Indeed into the 2026 period** without explicit justification; the Indeed subset has different extraction quality.

### Seniority operationalization

**Primary:** Combined best-available column.

```sql
CASE
  WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
  WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
  ELSE NULL
END AS seniority_best_available
```

**Required co-equal validator:** `yoe_extracted <= 2` proxy (and `<= 3` as alternative). Label-independent. If the combined column and YOE proxy disagree on direction, **investigate before picking a side** — this is how the AI/ML "less junior-heavy" artifact (T09) was caught in T28.

**Ablation set to report on every seniority-stratified finding:**
1. Combined best-available (primary)
2. `seniority_native` (arshkon-only baseline for entry analyses; do NOT pool asaniczka)
3. `seniority_final` (arshkon-only baseline for entry analyses; do NOT pool asaniczka)
4. `seniority_imputed` (where != unknown; text-only signal)
5. YOE-based proxy (`yoe_extracted <= 2` share)

**Asaniczka rule:** Asaniczka has zero native entry labels.
- Under `seniority_native` / `seniority_final`: use arshkon-only baseline. Do not pool.
- Under combined column, `seniority_imputed`, and YOE proxy: asaniczka can be included.

### Text source recommendation

**Primary text column:** `coalesce(description_core_llm, description_core, description)`. Use raw `description` (not cleaned) for any regex pattern that depends on preserved stopwords, phrases, or exact markdown (e.g., phrase matching like "lead a team", section-header detection, `c++` / `c#` detection).

**Text-source caveat for cross-period metrics:** 2024 is 91% LLM-cleaned, 2026 is 21% LLM-cleaned. Any text-derived metric that depends on LLM-vs-rule cleaning (length, token density, style markers) must report under both the aggregate corpus and the LLM-text-only subset. Length, em-dash density, and vocabulary diversity are known to be sensitive; credential stacking, AI mention rate, and scope breadth are NOT sensitive (V1 tested).

**swe_cleaned_text.parquet has English stopwords stripped.** Do not use it for phrase-level analyses — read raw text from `unified.parquet` instead. Documented in `shared/README.md` after the fact.

### Pattern recommendation

**Use T22's validated patterns at `exploration/artifacts/shared/validated_mgmt_patterns.json`** for any mentoring, hedge, firm, AI-tool, or scope-language computation. These were validated at ≥90% precision on 50-sample manual review. Any new pattern must be validated the same way (Section 14 — methodological recommendations).

**Specific pattern hazards (do not use the naive version):**
- `hire` / `hiring` / `recruit*` — ~8-28% precision (recruiter disclaimers, "hiring process", "contract-to-hire", HR boilerplate). Use the strict mentoring detector instead for senior-shift claims.
- Bare `agent` — ~55-72% precision (insurance agents, legal disclosures, robotics autonomy, "change agent" HR idiom). Use `agentic` (~95% precision) or `AI agent` / `multi-agent system`.
- `mcp` — matches "Microsoft Certified Professional". Removed from T22's `ai_tool` patterns.
- `\bc\+\+\b` — regex cannot match (silent failure). Use direct substring search on `' c++'` or `'c++,'` etc.
- `\bc#` / `\bcsharp\b` — also broken in the matrix; substring search works.

### Tech matrix caveat

**`swe_tech_matrix.parquet` has silent failures for c++ and c# (verified). Other tech columns were partially audited by V2 and found to be OK at the single-column level, but the matrix was not comprehensively re-validated.** Recommendation:
- For the headline "tech count" metric, **use V2's 39-tech detector on raw description** (mean 4.03 → 4.67, +15.8%; median 3 → 4; dedup mean +27%). Script: `exploration/scripts/V2/A_tech_detector.py`.
- For single-tech growth/decline claims, verify the matrix column against a direct LIKE query on raw description before citing the matrix number.
- Do NOT cite T11's "+34%", T14's "+60% median", or T19's "nearly flat" tech-count growth. Use the V2-reconciled number.

### Description hash dedup

The combined-column 2026 entry pool has ~23% exact-description_hash duplicate contamination from six companies (Affirm 25×, Canonical 22×, Epic 14×, Google 5×, SkillStorm 4×, Uber 4×). **The preprocessing dedup fix is in flight** — see `docs/preprocessing-dedup-issue.md`. In the meantime, apply **within-company description_hash dedup** to any entry-level analysis and to any content analysis with company-concentration sensitivity. The YOE proxy (n=4,022) has much lower duplicate leverage than the combined column (n=390), so it's the cleaner metric for aggregate entry-share claims until the fix lands.

---

## 4. Known confounders with severity assessment

| Confounder | Affects | Severity | Mitigation |
|---|---|---|---|
| **Native-label contamination** (41% arshkon entry YOE ≥5 vs 9.5% scraped) | All `seniority_native`/`seniority_final` entry comparisons; original "junior decline" headline | **SEVERE** | Use combined column primary + YOE proxy validator. Never use native labels alone for entry-level cross-period comparisons. Asaniczka rule: arshkon-only baseline for native/final comparisons. |
| **Text-source composition** (2024 91% LLM-cleaned vs 2026 21%) | Cross-period length metrics, style markers, em-dash density, vocabulary diversity. Does NOT affect credential stacking, AI mention, or scope breadth | **MODERATE** | Report aggregate + LLM-text-only subset side by side for any text-derived metric. Length growth is ~half real content, ~half text-source composition. |
| **Company concentration / duplicate templates** | Small-pool entry analysis (combined-col n=390 is 23% duplicates); open-ended corpus comparisons dominated by Amazon-style templates | **MODERATE** (fix pending) | Apply description_hash dedup. Prefer YOE proxy (larger pool). Use company capping (≤20/company) as primary spec for Fightin' Words and topic models. |
| **Tech matrix regex bugs** (c++, c# verified broken; possibly others) | Any claim that uses `swe_tech_matrix.parquet` columns for c_cpp, csharp. Possibly `rag` column has ~8% false positive rate from `leverage` contamination (V2 A2). Matrix was not comprehensively audited | **MODERATE** | Use V2's 39-tech detector for aggregate tech count. Verify single-tech claims against direct LIKE on raw description. Do not cite matrix numbers for c_cpp/csharp. |
| **Pattern precision** (naive `hire`/`agent`/`mcp` inflate 3-5×) | T11/T21 pattern-dependent absolute levels; any cross-task comparison using inconsistent patterns | **SEVERE for naive use; MITIGATED by T22 validated patterns** | Use `validated_mgmt_patterns.json`. Growth ratios are consistent across pattern variants (both ~2.5× for mentoring); absolute levels differ by 47-65% between T11 and T22 patterns. |
| **JOLTS hiring-cycle position** (arshkon at 87K hires trough; scraped cycle position unclear) | Within-2024 calibration of rate-of-change estimates; mid-senior mix may shift during trough | **MODERATE** | Limits within-2024 interpretation of temporal rate-of-change. Does not invalidate cross-period direction claims, but any "this is a 2024-2026 secular trend" claim should note the cycle-position caveat. |
| **Industry composition skew** (arshkon entry pool is finance/banking-skewed per T12; scraped has different mix) | Entry-level term comparisons with arshkon 2024 baseline; open-ended corpus comparisons using arshkon entry subset | **MODERATE** | Profile industry composition before cross-source entry-level term comparisons. The finance terms `laundering`, `vba`, `macros`, `econometric`, `liabilities` in T12's arshkon entry top terms are composition, not signal. |
| **Aggregator composition shift** (asaniczka 27.3% aggregator → scraped 14.3%) | Any aggregator-stratified metric; entry-share under sources without aggregator exclusion | **MINOR** (noted as sensitivity) | Aggregator-exclusion is in the standard sensitivity framework. V1 confirmed the 2026 entry rise survives aggregator exclusion. |
| **NYC scraper coverage 28× growth** (66 → 1,862 postings) | Metro-stratified comparisons for NYC specifically | **MINOR** (frame, not bug) | Flag NYC in metro tables; do not over-weight NYC in metro stratification. |
| **AI-benchmark selection bias** (StackOverflow survey respondents may be more AI-adopting than the worker pool) | Absolute gap size in RQ3 (employer 52% vs worker 80%) | **MINOR to MODERATE** | Report RQ3 gap with benchmark sensitivity bands. The direction (employer < worker) is unambiguous; the 24-28pp magnitude has uncertainty. |

---

## 5. Discovery findings

### Confirmed hypotheses

- **Scope/content expansion is real and universal across domains.** Credential stack ≥7: 3.8% → 20.5% (5.4×); robust to LLM authorship (T29), domain composition (T28: within every archetype, legacy stacks drove it more than AI/ML), text-source confound (V1: credential stacking survives where length does not). **Confidence: VERY STRONG.** Verified by T11, V1, T28, T29, V2 independently.
- **Senior tier reorganized to IC + mentoring + technical orchestration.** People-management collapsed (-58% density); mentoring doubled any-share (11% → 22%); technical orchestration grew +47% density with new vocabulary (`agentic` 415×, etc.). **Confidence: VERY STRONG.** T11 / T21 / T28 converge across four angles. Mentoring is "the single most robust cross-domain directional finding" — grew +1.9 to +29.4pp in every archetype with strict people-manager terms near-zero universally (T28).
- **Tech-cluster-wide restructuring (not SWE-specific).** SWE AI +29pp; SWE-adjacent +30.4pp (~105% of SWE per V2 broader pattern; T18 originally 83%); control +1.7pp. Length and scope growth LARGER in adjacent than SWE. **Confidence: VERY STRONG.** T18 + V2 reinforce.
- **AI/ML domain growth +11pp.** Verified independently by two proxies (title-based +11.86pp, strong-keyword body +13.69pp). Third method: T12 BERTopic +11.95pp. **Confidence: VERY STRONG.**
- **Entry-share rise is real.** Under combined column +3.48pp, imputed +4.66pp, YOE proxy +6.27pp. Robust to dedup, aggregator exclusion, top-20 removal, company capping (V1 sensitivity grid). **Confidence: STRONG for direction; magnitude depends on method.**
- **Mid-senior credential vocabulary stripping.** `qualifications`, `degree`, `bachelor`, `required`, `requirements` survive capping in T12 2024-favored top-30 (V1 F verified). **Confidence: STRONG.**
- **AI requirements are structurally hedge-heavy.** Hedge:firm ratio in 80-char AI windows is 10-11× in BOTH periods (V2: 10.61 → 11.59). Structurally stable. **Confidence: STRONG.**
- **Employer AI mention rate is BELOW worker AI usage.** Direct-only 52.9-55.8% (2026) vs ~80% StackOverflow. Gap of 24-28pp. Robust across four benchmark scenarios. **Confidence: STRONG (direction); MODERATE (exact gap size).**
- **Director↔mid-senior boundary blurred specifically.** AUC 0.751 → 0.686 (-0.065). Directors collapsed to mid-senior profile on people-management, mentoring, AI; only strategic scope distinguishes. **Confidence: MODERATE** (small samples; directors <2% of SWE).
- **AI-mentioning senior profile flipped 2024 → 2026.** 2024 AI seniors were mentor-heavy (research-mentor profile); 2026 are tech-orchestration-heavy (IC-builder profile). Same cohort, flipped positioning (T21). **Confidence: STRONG.**
- **Market organizes by tech domain, not seniority.** NMI archetype × language = 0.11 vs archetype × seniority = 0.015 (T09). No semantic seniority convergence (T15). Feature-based seniority boundaries mostly stable (T20). **Confidence: VERY STRONG.**

### Contradicted hypotheses (REJECTED or REFRAMED)

- **"Junior share is declining" — REJECTED.** Artifact of native-label contamination in arshkon. Under label-independent measurement (combined column, YOE proxy), entry share rose modestly. The original Gate 1 reframing stands.
- **"AI/ML is structurally less junior-heavy" — REJECTED (T28).** A routing-coverage artifact: ML-Engineer-style titles bypass the Stage 5 rule router, causing the combined column to under-represent AI/ML entry rows. Under YOE proxy, AI/ML 2026 entry share is 16.0% vs rest-of-market 17.2% — essentially identical.
- **"90% within-company junior decline" (T06) — WITHDRAWN.** Used contaminated native-label inputs. T16's corrected decomposition is the right version; see Section 12.
- **"Tech count grew 34-60%" — REFRAMED.** V2 reconciled to ~16-27% mean / 3→4 median. T11/T14 used the broken matrix; T19's "nearly flat" was text-source confounded.
- **"SWE-specific restructuring" — BROADENED.** The pattern is tech-cluster-wide. SWE-adjacent shows equal or greater content expansion.
- **"Anticipatory employer AI restructuring" (original RQ3) — INVERTED.** Employers lag workers by 24-28pp but are catching up 4-10× faster. The gap is closing, not opening.
- **"Recruiter LLM authorship is the mechanism" — REJECTED (T29).** Length, AI, and breadth headlines persist or strengthen in low-LLM-score subset. Cross-posting variance unchanged (2.25 → 2.19), contradicting "LLMs uniformize" prediction. Microsoft is the highest-scoring company in both periods, showing the score measures polished-marketing-prose not LLM authorship. **Qualification:** Credential stacking is partially attenuated in low-LLM (V2: +29% Q1 vs +149% full). The T29 claim "every headline persists or strengthens" is overstated for credential stacking specifically.
- **"Senior shift is people-management to AI orchestration" — REFRAMED.** The actual shift is to **IC + mentoring + technical orchestration**, with domain heterogeneity (mentoring DECLINED in AI/ML while growing everywhere else). Explicit people-management language is actively being stripped, not inflated.
- **"Semantic convergence between junior and senior postings" — REJECTED (T15).** Cosine similarity junior↔senior is flat or slightly diverging (emb -0.001; TF-IDF -0.022); below within-2024 noise by 4×. Holds under LLM-text-only subset (V1 E). Not a text-source artifact.

### New discoveries (not in original RQ1-RQ4)

- **The AI-senior profile flip within the same cohort.** 2024 AI-mentioning seniors were mentor-heavy (research-mentor profile); 2026 are technical-orchestration-heavy (IC-builder profile). Early AI experts were positioned as researchers/mentors; current AI experts are positioned as IC builders. Surfaced by T21. **Novelty: HIGH.** Sharpest single Wave 3 finding.
- **The defense-contractor over-representation in 2026 entry postings.** SpaceX 65% own-entry share, plus Northrop, Raytheon, Peraton, Booz Allen, Leidos over-represented in YOE-proxy top contributors. Surfaced by V1 + T16. **Novelty: HIGH.** Raises a qualitative research question about defense contractors maintaining new-grad pipelines while other employers pulled back.
- **Mentoring DECLINED in AI/ML while it grew everywhere else.** Senior shift in AI/ML is purely tech-orchestration; in non-AI domains, it's mentoring + tech-orchestration. Surfaced by T21 + T28. **Novelty: HIGH.** The only cross-domain heterogeneity in an otherwise universal finding.
- **AI/ML is becoming LESS concentrated, not more.** Top-20 concentration in AI/ML fell 31% → 22% (T28). AI/ML is diffusing into mainstream tech employers' stacks (Microsoft, JPMC, GM, Uber posting AI/ML roles), not consolidating into AI specialists.
- **Direct employers are MORE ghost-like than aggregators** on content measures (T22). Kitchen-sink scores, any_ai, agentic are higher in direct-employer postings. The naive "aggregators are ghost-post heavy" intuition is wrong.
- **90.4% of scraped companies with ≥5 SWE postings have ZERO entry rows** under the combined column; 49.3% under YOE proxy (T16). Entry posting is a specialized activity at a small set of new-grad-program employers, not a market-wide function.
- **Within-title entry share exploded for "software engineer" specifically.** 1.2% → 15.3% for the exact title string "software engineer" (T10). 12× more entry-labeled under the same string — supports the labeling-explicitness story.
- **The structural 10× AI hedge ratio** (T22). AI requirements have always been ~10× more hedged than non-AI requirements (in both 2024 and 2026). This is a fact about how employers signal AI demand, not a temporal shift. Combined with the AI mention quintupling, it produces a composition-driven rise in the aggregate aspiration ratio.
- **Employer labeling explicitness increased** (+10.6pp known-rate jump in `seniority_imputed`; junior keyword share nearly doubled; "entry" keyword in title 3×). Surfaced by the seniority deep-dive. A separate finding about how the labor market data ecosystem evolved alongside the labor market itself.
- **Tech density (per 1K chars) FELL 26% even as tech count rose.** Postings grew longer faster than they packed requirements (T11 / T14). Originally flagged as the trigger for the LLM-authorship hypothesis, which was then cleanly rejected by T29 / V2. The density decline is real but doesn't map to any single mechanism yet.

### Unresolved tensions

- **Credential stacking low-LLM partial attenuation.** V2 found credential stacking grew +29% in Q1 (low-LLM) vs +149% in the full corpus. The hypothesis rejection holds for length and AI growth (both persist robustly in Q1), but credential stacking is partially mediated by writing style. **Unresolved:** Is this a real writing-style effect (employers who write in LLM-style ALSO list more credentials), or a measurement artifact from the cruder 7-category proxy V2 used? Needs a clean re-run with T11's exact stack definition on the V2 Q1 subset.
- **The tech-count discrepancy was resolved by V2 but raises a matrix-wide audit question.** V2 reconciled to ~16-27% mean / 3→4 median. But V2's audit also noted the `rag` matrix column has ~8% false positive rate from `leverage` tokens, and did not do a comprehensive audit of all 153 matrix columns. **Unresolved:** Are other matrix columns silently broken beyond c++/c#?
- **T16's 87% between-company is convention-dependent.** Exact replication under arshkon-only min≥3 (V2 B). But drops to 0.5% within at min≥5 and flips to ~50/50 under pooled-2024 panel. **Unresolved:** What is the "right" panel convention? Arshkon-only is defensible (asaniczka has no native entry labels) but the finding is narrower than the clean 87% headline suggests.
- **Text-source confound asymmetry.** The text-source composition shift explains ~half of length growth but does NOT explain credential stacking, AI mention, or scope breadth growth. **Unresolved:** Why the asymmetry? One possibility: LLM-cleaned text is tighter and shorter, so length is directly composition-driven, but credential-listing is a discrete count that isn't affected by text compression. Needs a clean diagnostic comparing LLM-only credential stack growth to aggregate credential stack growth.
- **T18 DiD sign flip on SWE-adjacent.** T18 reported DiD SWE-adjacent = +3.9pp (SWE slightly ahead); V2 reported -1.4pp (adjacent slightly ahead). Both within noise. The qualitative conclusion is identical (adjacent ≈ SWE), but the sign flip means the data cannot tell us whether SWE or adjacent is leading/lagging. **Unresolved.** May be inherent to the snapshot structure.

---

## 6. Posting archetype summary (T09)

T09 ran BERTopic on a balanced 8K sample and produced 23 archetypes. Key findings:

- **NMI archetype × primary_language = 0.11; × period = 0.04; × seniority = 0.015; × YOE junior = 0.008.** Primary language dominates the embedding 5× over period and 7× over seniority. **The natural latent structure of the SWE posting market is technology domain, not career level.** This validity-backbone finding reshapes the entire RQ framing.
- **AI/ML archetype grew +10.96pp** — the only large grower. It absorbs share from .NET (-3.5), frontend/web (-2.3), Spring/Java (-1.8), data engineering (-1.1), python/django (-0.9). T12's independent BERTopic run on a different sample gave +11.95pp — convergent.
- **T09's "AI/ML is less junior-heavy" was REJECTED by T28** as a routing-coverage artifact. Under YOE proxy, AI/ML entry share matches the rest of the market.
- **Archetype labels were propagated to 63K rows via nearest-centroid** (87% holdout accuracy) — saved as `swe_archetype_labels.parquet`. Use for any within-domain stratification in the analysis phase.

The 23 archetypes plus the +11pp AI/ML growth are the natural within-domain stratification unit for the analysis phase. All Wave 3 scope-inflation findings hold within every archetype (T28: breadth +17-50%, tech count +4-107%, scope count +41-153%, credential stack ≥7 ratio 1.5× to 64× depending on archetype). The +1.5× was AI/ML (smallest relative increase, because it already had the highest 2024 baseline at 25.7%); the +64× was ServiceNow (legacy stack). **Legacy stacks drove the credential-stack headline jump MORE than AI/ML** — a direct reversal of the naive "AI is the cause" intuition.

---

## 7. Technology evolution summary (T14 + V2)

**Core finding:** AI cluster fusion. In 2024, modern-AI tools (Copilot, Claude/OpenAI APIs, Hugging Face, RAG, LangChain) formed isolated 2-4 node communities. In 2026 they fuse with traditional ML (PyTorch, TensorFlow, pandas, NLP) into a 29-member ML+GenAI mega-community. AI-mentioning posting share grew 2.8% → 26.2% (9.4×) on T14's count; 13.7% → 52.0% under the V1 union of 11 AI terms.

**Tech count growth (reconciled by V2):** **~16-27% mean, median 3→4.**
- T11 reported +34% mean — broken matrix, not wrong arithmetic but wrong measurement.
- T14 reported 5→8 median (+60%) — same broken matrix.
- T19 reported nearly flat 2.49→2.73 — narrow safe-15 list on LLM-coalesced text, which is itself composition-shifted. Broken c++/c# regex amplified the flatness.
- V2 independent 39-tech detector on raw description: mean 4.03 → 4.67 (+15.8%); median 3 → 4 (+33%); dedup mean +27%.

**New vocabulary (absent or near-zero in 2024, measurable in 2026):** `agentic` 415× (0 → 8% corpus share per T23), `rag` 76×, `langchain` 30×, `mcp` 29× (valid when restricted to the tool use; 'MCP' bare is contaminated), `prompt engineering` 20×, `guardrails` 15×, `multi-agent` 40×, `copilot` 4.2%, `claude` 350×, `cursor` 604× (tool).

**Legacy stack decline (V2 confirmed):** SQL -4.9pp, JavaScript -3.9pp, Linux -2.5pp, .NET -2.1pp. These are genuine declines under V2's independent detector on raw description.

**Growers (V2 confirmed):** Python +14.8pp, AWS +6.2, Git +5.8, Kubernetes +5.2, Terraform +5.2, TypeScript +4.8, React +4.4, GCP +4.3, LangChain +3.5 (30×), PostgreSQL +3.5, PyTorch +3.4 (2.8×), **C++ +3.2 (after V1 fix; T14 originally reported C++ as declining due to the regex bug)**.

**The c++/c# bug story for posterity.** The `\bc\+\+\b` regex cannot match because `\b` (word boundary) does not lie between `+` and end-of-word. Scraped LinkedIn text also contains markdown-escaped variants (`C\+\+`). The buggy matrix reported C++ at 0.5% mention rate; actual is 19.0%. Same issue for C#. T14 reported "5 declining technologies" including C++ and C#; both must be removed from the decliner list. C++ actually grew +21% and is a top-10 systems growth story (co-occurs with embedded, linux, rust per V1). This is the canonical example of the silently-broken-regex failure mode and should be cited in the paper's methods contribution.

---

## 8. Geographic heterogeneity summary (T17)

26 metros with ≥50 SWE postings in both periods. Key patterns:

- **AI↔scope metro correlation +0.43 (p=0.03).** Metros with larger AI gains also have larger scope-language gains. The content expansion travels with AI.
- **AI↔entry is uncorrelated (r=-0.09, p=0.65).** AI surge and entry-share rise are **geographically decoupled** — different mechanisms operating in different places.
- **Hubs lead non-hubs modestly:** +3pp AI, +5pp scope. Not a dominant effect.
- **AI gains top metros:** Tampa +30pp, Salt Lake City +27pp, Miami/NYC/SF +26pp. Lagging: Detroit/DC/LA/Denver at +11-12pp.
- **Entry (YOE) top metros:** San Diego +26pp (Qualcomm effect — 64% own-entry share), SF Bay / Seattle +12pp.
- **Remote share 0% → 22.3% corpus-wide.** Remote and onsite have identical AI/scope rates — no remote premium on AI.
- **AI/ML archetype concentration:** SF 26.7%, Boston, Seattle, and **Houston at rank 4 (15.6%)** — the Houston ranking is a surprise worth investigating qualitatively (is this energy-sector AI adoption?).
- **NYC scraper coverage grew 28×** (66 → 1,862 SWE). This is frame, not bug, but any NYC-specific claim should be flagged.

The geographic data supports metro-stratified analysis for AI content, scope, and remote variables; it does NOT support strong metro-level causal claims because of the snapshot structure and NYC coverage change.

---

## 9. Senior archetype characterization (T21 + T28)

The senior-tier restructuring is the sharpest paper-quality narrative finding. Four convergent lines of evidence.

**T21 density metrics (senior-only subset):**
- People-management density: **-58%** (the collapse)
- Mentoring density: **+33%** (any-share 11% → 22%)
- Technical orchestration density: **+47%** (any-share 37% → 62%; biggest single shift)

**T21 k-means cluster shift:**
- People-Manager cluster: 3.7% → 1.1% (**-70% relative**) — formal people-management archetype is shrinking.
- Mentor-heavy cluster: 8.6% → 10.8% (+26%)
- TechOrch-heavy cluster: 5.0% → 7.6% (+52%)

**New AI-orchestration vocabulary (near-zero in 2024):**
- `agentic` 415×
- `prompt engineering` 20×
- `guardrails` 15×
- `multi-agent` 40×
- plus `langchain`, `langgraph`, MCP-tool (not bare MCP), `rag`

**T28 cross-domain stratification (the most robust cross-domain finding in the entire exploration):** Mentoring grew +1.9 to +29.4pp in every archetype, with strict people-manager terms near-zero universally (17/20 archetypes). The growth is truly universal — EXCEPT for AI/ML, where mentoring DECLINED and the senior shift is purely toward technical orchestration. Backend/Enterprise is the only archetype where people-management density actually rose.

**Director-specific finding (T20 + T21):** The AUC 0.75 → 0.69 boundary blur between mid-senior and director happened not because the tiers are becoming indistinguishable everywhere, but because **directors collapsed onto the mid-senior profile on every dimension except strategic scope**. Directors stopped being a separate people-management track. Mentoring drifted DOWN from director to mid-senior; people-management dropped out of top discriminators for director. This is a tier-specific merge, not market-wide blurring.

**The AI-senior profile flip (T21):** The 2024 cohort of AI-mentioning seniors were mentor-heavy (research-mentor profile); the 2026 cohort are technical-orchestration-heavy (IC-builder profile). Same group, profile flipped within two years.

**Domain heterogeneity — the most interesting unexplained pattern:**
- Frontend, Backend, Cloud: mentoring GREW
- AI/ML: mentoring DECLINED; senior shift is purely tech-orchestration
- Backend/Enterprise: only domain where people-management density actually ROSE

The AI/ML exception needs qualitative interpretation in the analysis phase. Hypothesis: AI/ML seniors in 2024 were senior researchers (mentor-heavy profile); in 2026 they've been joined by AI-orchestration specialists with a different profile, so the aggregate shifts without individual-level changes. Or: AI/ML teams are flatter and the mentoring layer was always thinner. This is a clean interview target.

---

## 10. Ghost / aspirational prevalence (T22)

**The structural 10× AI hedge ratio (T22, V2 verified):** Within 80-character windows around AI terms, hedge:firm marker ratio is **10.61 (2024) → 11.59 (2026)** — stable across periods. Global baseline ratio: 2.03 → 2.69. **AI requirements have always been ~4-5× more hedge-heavy than non-AI requirements**, and this is NOT a 2024-2026 shift. What changed is that AI mention prevalence quintupled (14.3% → 51.2%), so the aggregate corpus-level aspiration ratio rose as a composition effect.

**Kitchen-sink score (count of simultaneous demand categories):** Kitchen-sink ≥12 share tripled (entry 3.7% → 12.6%; mid-senior 7.7% → 27.3%). Aspiration-ratio >1 share: entry 23.7% → 33.6%, mid-sen 16.4% → 26.8%.

**Template saturation caveat:** Mean pairwise cosine similarity within companies rose 0.604 → 0.669 after exact-hash dedup; dedup halved the saturated-company share (11.9% → 5.8%). This confirms V1's six-company duplicate artifact. **The dedup fix (see preprocessing-dedup-issue.md) will reduce but not eliminate template saturation** — some companies post near-identical variants that share content but differ in template metadata.

**Direct employers are MORE ghost-like than aggregators** (T22). Kitchen-sink, any_ai, agentic are higher in direct-employer postings. The naive "aggregators run more ghost posts" intuition is wrong.

**Entry-level ghost analysis recommendation:**
- **Combined column:** Use for AI-content ghost analysis (surfaces real new-grad postings with dense stacks — IBM intern, GM intern, ByteDance grad).
- **YOE proxy:** Use for aggregate share counts (surfaces senior roles mis-captured by "2+ yrs" phrasing — Visa Staff SWE, CVS Senior, Xylem Senior AI).
- **The two top-20 ghost lists have ZERO overlap** (T22). They measure different things. Do not conflate.

**Validated patterns saved:** `exploration/artifacts/shared/validated_mgmt_patterns.json`. All patterns validated at ≥90% precision on 50-sample manual review. MCP removed from `ai_tool` due to Microsoft Certified Professional contamination.

---

## 11. Cross-occupation findings (T18 + V2)

**The validity backbone for the tech-cluster-wide framing.**

**AI rate by group (V2 broader pattern):**
- SWE: 13.45% → 42.42% (+28.97pp)
- SWE-adjacent: 11.57% → 41.93% (+30.36pp) — **slightly exceeds SWE**
- Control: 0.36% → 2.05% (+1.69pp)

**DiDs:**
- SWE − control: **+27.28pp** (strong)
- SWE − adjacent: **-1.39pp** (adjacent slightly ahead, within noise)

**T18 originally reported adjacent at 83% of SWE magnitude (+19.0 vs +22.9); V2 broader pattern finds adjacent at ~105% (+30.4 vs +29.0). The qualitative conclusion is identical but stronger under V2: the pattern is occupational-tech-cluster-wide, not SWE-specific.** The "tech-cluster-wide" framing should be stated with high confidence in the analysis phase.

**Length and scope growth larger in adjacent than SWE** (T18 text-source-controlled):
- Length: adjacent +629 vs SWE +527 chars
- Scope language: adjacent +26.3pp vs SWE +19.0pp

**SWE↔control CONVERGED** (cosine +0.079). All white-collar postings drift toward shared longer/inclusive style. **SWE↔adjacent slightly DIVERGED** (-0.022) — opposite of a naive boundary-blurring prediction.

**"AI Engineer" within-title evolution** (the clean paper-figure finding):
- 2024: n=20, 0% agentic, 40% PyTorch, 15% LLM (pytorch/ML role)
- 2026: n=321, 45% agentic, 32% PyTorch, 54% LLM (LLM-agentic role)
- 14-16× volume growth
- Within-title content evolved from traditional ML to LLM-orchestration, with no naming change.

**Embedding-adjacent-only subset sensitivity:** On the stricter `swe_classification_tier='embedding_adjacent'` subset (n=3,166 in 2026), AI rate grew +28.6pp — nearly identical to full SWE. The adjacent finding holds under the stricter tier definition.

---

## 12. Entry-share story arc (the full reframing)

This is the three-step reframing that took the most effort during the exploration. Stating it in full for clarity.

**Reframing 1: Gate 1 — The "junior decline" headline was an artifact.**
- Original native-label finding: arshkon 22.3% entry → scraped 13.7% entry (-8.7pp). Reported by T01-T07.
- Seniority deep-dive found: 41% of arshkon `seniority_native='entry'` SWE rows have YOE ≥ 5, vs only 9.5% in scraped. The arshkon entry pool has a bimodal YOE distribution with a spurious 5+ year cluster inflating the 2024 baseline by ~40%.
- Under label-independent methods: YOE ≤ 2 proxy 15.0% → 16.6% (+1.6pp); `seniority_imputed` 6.8% → 6.7% (flat); explicit entry signals in text 2.1% → 2.6% (+0.5pp).
- Verdict: entry share was approximately stable or modestly increased, not declining. Original headline withdrawn.

**Reframing 2: Gate 2 / V1 — The entry rise is real.**
- Under combined best-available column: 1.87% → 5.35% (+3.48pp).
- Under `seniority_imputed`: 2.01% → 6.66% (+4.66pp).
- Under YOE ≤ 2 proxy: 10.34% → 16.61% (+6.27pp).
- Robust to dedup, aggregator exclusion, top-20 removal, company capping (V1's eight-variant sensitivity grid). Only reverses in one narrow cell (arshkon-only × YOE × cap20 at -0.79pp, within noise). V1 could not reproduce T08's dramatic reversal claim.
- V1 also flagged that ~23% of the 2026 combined-col entry pool is description_hash-duplicate postings from six companies (Affirm, Canonical, Epic, Google, SkillStorm, Uber). The YOE proxy (n=4,022) has much lower duplicate leverage than the combined column (n=390) and is the cleaner metric for headline claims.
- V1 also discovered: 91% of scraped companies with ≥5 SWE postings have ZERO entry rows under the combined column. Entry posting is a specialized activity.

**Reframing 3: Gate 3 / T16 — The rise is composition, not within-company pivot.**
- Under T16's arshkon-only min≥3 overlap panel (n=206 companies): YOE entry rise +6.50pp total, decomposed as +0.82pp within-company + **+5.68pp between-company = 87% between-company.**
- **Combined-column within-company Δentry is -0.27pp** (slightly negative). The same firms are not pivoting toward junior hiring; the market mix is shifting under them.
- 2026 reweights toward companies that were ALREADY entry-heavy — Google, Walmart, Qualcomm, SpaceX, Amazon, Microsoft, Cisco, Visa, and an unexpectedly over-represented defense-contractor cluster (SpaceX 65% own-entry share, plus Northrop, Raytheon, Peraton, Booz Allen, Leidos).
- T16's 90.4% concentration finding: 90.4% of scraped companies with ≥5 SWE postings have ZERO entry rows under the combined column; 49.3% under YOE proxy. Entry-level posting is a specialized activity, not a market-wide function.
- **T28 killed the "AI/ML is less junior-heavy" intermediate finding** — it was a routing artifact. Under YOE proxy, AI/ML entry share 16.0% vs rest 17.2%.

**V2 panel-convention caveat (the current best statement):**
- Under T16's arshkon-only min≥3 convention: within +0.82pp, between +5.68pp (87% between) — replicates exactly.
- Under arshkon-only min≥5: within +0.03pp (0.5% within; ~100% between).
- Under arshkon-only min≥10: within -1.37pp (between exceeds total).
- Under pooled-2024 min≥3: within +4.30pp, between +4.76pp (~47% within).
- Under pooled-2024 min≥5: within +5.70pp, between +3.60pp (~61% within).
- **The qualitative finding — "between-company composition explains a substantial majority of the entry rise" — holds across panel definitions. But the specific 87% headline is arshkon-only convention-dependent. The pooled-2024 panel gives ~50%.**

**The honest current best statement:**
> Between-company composition explains a majority of the entry-share rise (50-87% depending on panel definition), driven by 2026 reweighting toward employers with formal new-grad programs — particularly a defense-contractor cluster. Within-company change explains the remainder (essentially zero under the arshkon-only convention, ~40-60% under pooled-2024). Under the combined best-available column, within-company change is -0.27pp (slightly negative) regardless of panel — the same firms are not pivoting toward junior hiring. 90.4% of scraped companies with ≥5 SWE postings have ZERO entry rows under the combined column, confirming that entry posting is a specialized activity at a small set of new-grad-program employers.

This is a more honest (and more interesting) story than "entry share rose" or "entry share declined."

---

## 13. New hypotheses from T24 (placeholder)

T24 has not yet produced output as of this writing (2026-04-10). When T24 is complete, its hypothesis list should be inserted here verbatim. In the meantime, based on the consolidated state of play above, the highest-priority follow-up hypotheses for the analysis phase are:

1. **The AI-senior profile flip within the same cohort.** Test whether 2024 AI-mentioning seniors (mentor-profile) and 2026 AI-mentioning seniors (IC-builder profile) are the same people at different career moments, or different selection cohorts. Requires within-company panel linkage.
2. **The defense-contractor entry-pipeline hypothesis.** Why are SpaceX, Northrop, Raytheon, Peraton, Booz Allen, Leidos over-represented in 2026 entry postings? Did they expand new-grad programs, or did other employers contract? Test via the arshkon 2024 presence of these companies (V1 noted most had zero arshkon entry rows — a gap to investigate).
3. **The AI/ML mentoring-decline heterogeneity.** Why did mentoring decline in AI/ML while growing everywhere else? Two candidate mechanisms: (a) AI/ML teams are flatter and the senior-mentor layer was always thinner; (b) 2024 AI/ML seniors were research-mentor archetypes being diluted by 2026 tech-orchestration specialists. Test with within-archetype role composition changes.
4. **The lagged employer AI catch-up rate.** At current growth rates (employer 4-10× faster than worker), when do employer AI requirements catch up to worker AI usage? Extrapolate; test bounds under multiple benchmark scenarios. This is a substantive forecasting question.
5. **The legacy-stack credential explosion.** Why did ServiceNow (64×), Spring/Java (23×), .NET (18×) drive the credential-stack jump MORE than AI/ML? Candidate: legacy-stack employers are running harder to compete in a consolidating market, signaling maximum requirements to attract scarce candidates. Or: legacy-stack postings are from larger/more formal employers whose writing style has always been more credential-heavy. Test with employer-size and formality controls.
6. **The within-title explosion of "software engineer" entry share (1.2% → 15.3%).** Is this employers relabeling mid-level roles as "software engineer" instead of "senior software engineer"? Or is this a genuine increase in new-grad postings under the generic title? Within-company "same title, changed seniority" tracking.
7. **The tech-cluster-wide framing vs SWE-specific:** Given that adjacent shows equal or greater content expansion, is SWE the leading or lagging indicator, or are they contemporaneous? The snapshot data cannot answer this directly but within-employer cross-occupation comparisons might.

Once T24 returns, replace this section with its ranked list.

---

## 14. Method recommendations for the analysis phase

**Within-vs-between decomposition is mandatory for any aggregate trend.** Entry share, AI mentions, length, and tech count all have different mechanisms when decomposed:
- Entry share: 50-87% between-company composition (convention-dependent).
- AI mentions: ~91% within-company (T16).
- Length: about half within-company, half text-source composition.
- Tech count: mostly within-company (T16: 113%), but sensitive to matrix bug.

Any aggregate headline that does not distinguish these mechanisms will be misinterpreted. Report within-vs-between for every aggregate trend in the paper.

**Cross-occupation control is a default sensitivity, not an afterthought.** The SWE-specific framing should be tested explicitly on every content finding. T18's occupational DiD is the template. If a finding is SWE-only (e.g., if credential stacking holds in SWE but not in SWE-adjacent), that's interesting in its own right. If it holds cluster-wide (as most findings do), report the broader framing.

**Confounder ruling discipline.** Before claiming any finding is real, test at minimum: (a) domain composition (T28 stratification), (b) LLM authorship (T29 low-LLM Q1 subset), (c) text-source composition (LLM-only subset), (d) native-label contamination (combined column + YOE proxy ablation), (e) company concentration (cap ≤20 + dedup). The scope-inflation finding survived all five; that's why it's the strongest empirical signal. Any new finding that does NOT survive all five should be reported with the failing ablation explicitly called out.

**Pattern validation discipline.** For any new regex or keyword indicator:
1. Sample 50 matches per pattern per period; manually assess precision. Target ≥90%.
2. If the pattern uses a section classifier, report match distribution by section. A pattern that fires mostly in benefits/legal is structurally different from one that fires in requirements/responsibilities.
3. Cross-check against an independent method (structured column, direct LIKE, alternative extractor).
4. Canonical failure modes to pre-empt: `\b` near non-word characters (c++, c#, .net, node.js); polysemous tokens (`agent`, `hire`, `mcp`); acronym contamination (MCP vs Microsoft Certified Professional).
5. Save validated patterns to `exploration/artifacts/shared/` with a precision note in the JSON metadata.

**Test-driven development for non-trivial logic.** Assert statements at the top of scripts (V2's A_tech_detector.py is the template). The c++/c# regex bug would have been caught by a one-line assertion `assert pattern.search("c++")`. Any regex that processes special-character tokens must have positive and negative assertions before it runs at scale.

**Do not reload shared artifacts uncritically.** `swe_tech_matrix.parquet` has known silent failures; `swe_cleaned_text.parquet` has stopwords stripped. Shared artifacts are starting points, not ground truth. Verify every non-trivial use against raw data.

---

## 15. Sensitivity requirements

Findings requiring specific robustness checks in the analysis phase:

| Finding | Required sensitivity | Minimum variants |
|---|---|---|
| Credential stack ≥7 jump (5.4×) | Domain stratification; LLM-Q1 subset; text-source-controlled subset; capped corpus | 4 |
| Senior mentoring growth (11% → 22%) | Domain stratification (critical: AI/ML heterogeneity); T22 pattern vs T11 pattern; director-vs-mid-senior split | 3 |
| Technical orchestration growth | Cluster-based (T21) vs density-based (T11) | 2 |
| Entry-share rise (combined column, YOE proxy) | Panel convention (arshkon-only vs pooled-2024); dedup; cap20; aggregator exclude; within-vs-between decomposition | 5 |
| Length growth (+88% core) | Text-source LLM-only subset | 1 (mandatory) |
| Tech count growth | V2 39-tech detector on raw description as primary; matrix-raw as sensitivity only | 2 |
| AI mention rate (13.7% → 52.0%) | Union pattern vs `\bai\b` alone; LLM-Q1 subset; direct vs aggregator | 3 |
| Cross-occupation DiD | Embedding-adjacent strict subset; broader vs narrower AI pattern; SWE-adjacent vs control | 3 |
| Employer-worker AI gap (24-28pp) | Four StackOverflow benchmark scenarios; direct vs aggregator; narrower ai_tool vs broader any_ai | 4 |
| 10× AI hedge ratio | Two-period structural stability check; narrower vs broader hedge/firm pattern | 2 |
| T16 87% between-company | Panel convention (arshkon min≥3, min≥5, min≥10; pooled min≥3, min≥5); dedup | 6 |
| Director↔mid-senior blur | Feature ablation; combined column vs native; small-sample bootstrap | 3 |
| AI-senior profile flip | Within-cohort vs cross-cohort; T22 pattern vs T21 pattern | 2 |

---

## 16. Interview priorities (linking to T25)

T25 has not yet produced its artifact list as of this writing. The interview protocol update should focus on:

**Highest priority (new / sharpened findings from Wave 3):**
1. **The senior IC + mentoring + technical orchestration shift.** Ask specifically: "Has your team started asking senior engineers to mentor juniors more than two years ago? Have formal people-manager responsibilities (direct reports, performance reviews) become less common in senior IC job descriptions?" Probe whether this is real change or template language.
2. **The mentoring-declined-in-AI/ML heterogeneity.** Ask AI/ML practitioners specifically: "Does your team have a senior-mentoring layer? Has that changed over the past two years? Are AI/ML seniors doing more individual-contributor orchestration work and less mentoring than they used to?"
3. **The lagged employer AI catch-up.** Ask hiring managers: "When did your team start including AI tools (Copilot, Cursor, LangChain) in job descriptions? Why then? Do candidates actually need those tools on day 1, or is the inclusion aspirational?"
4. **The scope inflation (credential stack 5.4×).** Ask recruiters AND hiring managers: "Your 2026 entry-level postings list more requirements than 2024. Is this 'we ask for everything and filter later' or 'we genuinely raised the bar'? Can a 2026 entry hire actually do all the things the JD lists?"
5. **The composition-driven entry rise.** Ask specifically at the defense-contractor cluster (SpaceX, Northrop, Raytheon, Peraton, Booz Allen, Leidos): "Did you maintain or expand your new-grad program between 2024 and 2026? Did you see other employers pulling back? How did that affect your applicant pool?"

**Second priority (validity probes):**
6. **The labeling-explicitness shift.** Ask: "Did your company change how it labels entry-level postings between 2024 and 2026? Is 'software engineer' now more likely to mean new-grad than it used to?"
7. **The director merge.** Ask directors specifically: "Have your day-to-day responsibilities changed? Are you doing more IC work, more strategic scope, and less direct people-management than two years ago?"

**T25's artifacts** (when produced) should be used as the visual anchor for interview kickoff, not regenerated. Point to existing paths in `exploration/figures/` (specifically T21 for senior archetype shift, T18 for cross-occupation, T16 for company decomposition, T22 for hedge ratio, T23 for employer-worker gap).

---

## 17. Open methodological questions

1. **The tech matrix beyond c++/c#.** V2 audited a handful of columns and found the matrix is mostly clean at the single-column level, with one exception (`rag` column has ~8% false positive rate from `leverage` tokens). A comprehensive audit of all 153 matrix columns was not performed. **Needs:** Systematic re-validation or regeneration of the matrix using the V2 assertion-based pattern validation methodology. Until then, any single-tech claim using the matrix must be cross-verified against direct LIKE on raw description.

2. **The credential-stacking partial attenuation in low-LLM (V2 F).** The T29 hypothesis rejection holds for length and AI rate, but credential stacking is +29% in Q1 vs +149% in full corpus. Is this a real writing-style mediation (employers who write in LLM-style also list more credentials), a measurement artifact from V2's cruder 7-category proxy vs T11's exact stack definition, or a partial-mediation effect (writing style and credential-listing are correlated but distinct mechanisms)? **Needs:** A clean re-run using T11's exact stack definition on the V2 Q1 subset, plus a mediation-analysis-style decomposition.

3. **The convention-dependence of T16's 87%.** What is the "right" panel definition for the within-vs-between decomposition? Arshkon-only min≥3 is defensible (asaniczka has no native entry labels, so pooling dilutes the 2024 comparison) but not uniquely correct. Pooled-2024 min≥3 gives ~47% within. Arshkon-only min≥5 gives ~0% within. **Needs:** A principled choice of panel convention (pre-registered for the paper) and a sensitivity-bands presentation rather than a single point estimate.

4. **The text-source confound asymmetry.** Why does text-source composition explain half of length growth but none of credential stacking, AI mention, or scope breadth growth? Hypothesis: LLM-cleaned text is tighter and shorter, so length is directly composition-driven, but discrete count metrics (credentials, AI mentions, scope terms) are insensitive to text compression. **Needs:** A clean diagnostic comparing LLM-only credential stack growth to aggregate credential stack growth, plus the same for AI mentions and scope breadth. If the asymmetry holds, it's a methodological finding worth the paper's attention.

5. **The within-title "software engineer" explosion (1.2% → 15.3% entry).** Same exact title string, 12× more entry-labeled. Is this (a) employers relabeling mid-level roles as "software engineer" instead of "senior software engineer", (b) genuine increase in new-grad postings under the generic title, or (c) an artifact of the labeling-explicitness shift showing up in the title field? **Needs:** Title-level within-company panel analysis to disentangle.

6. **The defense-contractor over-representation.** V1 noted most 2026 top-20 entry contributors have zero arshkon presence. Is this a real expansion of defense-contractor new-grad pipelines, or a scraper-coverage change? **Needs:** Cross-check against BLS defense-sector posting volume and/or industry-specific job board data.

7. **The JOLTS cycle-position confound.** Arshkon was at a hiring trough (87K); scraped cycle position is unclear. T19's within-2024 rate-of-change estimation is the closest we have but is limited by discrete snapshots. **Needs:** Macro-control sensitivity on any rate-of-change claim. Does not invalidate cross-period direction claims.

8. **The volunteer-job contamination in asaniczka (T15 noted).** Low prevalence; doesn't affect aggregate findings; accepted as classifier imperfection. Flag for completeness; not a blocking issue.

---

## Appendix: canonical citations

When citing specific numbers in the analysis phase, use these verified sources as the canonical version:

- **Credential stack ≥7 jump:** 3.8% → 20.5% (T11); verified 2.85% → 15.97% (V1) = 5.4-5.6×. Use T11's rounded "5.4×" headline; V1 confirms within 5%.
- **AI mention rate:** 13.7% → 52.0% (union of 11 AI terms, V1 verified); `\bai\b` alone 6.8% → 43.6%.
- **AI/ML archetype growth:** +10.96pp (T09), +11.95pp (T12), +11.86pp (V1 title proxy), +13.69pp (V1 strong-keyword body proxy). Use "+11-14pp" band; report T09's number as canonical.
- **Entry-share rise:** combined column +3.48pp; `seniority_imputed` +4.66pp; YOE proxy +6.27pp (all V1 verified).
- **Length growth (core sections):** +88% aggregate; +26% under LLM-text-only subset; use both.
- **Tech-count growth:** V2 39-tech detector mean +15.8% (dedup +27%); median 3 → 4. Do NOT cite T11/T14/T19.
- **Senior people-management decline:** -58% density (T21); formal People-Manager cluster -70% relative (T21).
- **Senior mentoring growth:** 11% → 22% any-share (T21); +1.9 to +29.4pp in every archetype (T28) except AI/ML where it declined.
- **Senior technical orchestration growth:** +47% density (T21); any-share 37% → 62%.
- **Director↔mid-senior boundary:** AUC 0.751 → 0.686 (T20) under seniority_final.
- **Cross-occupation AI growth:** SWE +28.97pp, adjacent +30.36pp, control +1.69pp (V2 broader pattern). T18's 83% adjacent magnitude is superseded by V2's ~105%.
- **10× AI hedge ratio:** 10.61 (2024) → 11.59 (2026) in 80-char AI windows (V2). Global baseline 2.03 → 2.69.
- **Employer AI rate:** direct-only 14.5% → 55.8% (V2 broader pattern) or 11.2% → 52.9% (T23 narrower). Delta is +41pp either way.
- **Employer-worker AI gap:** 24-28pp, employer below worker. Use "~24-27pp" in text.
- **T16 between-company share:** 87% under arshkon-only min≥3 (exact replication, V2 B). Report with panel-convention caveat.
- **91% company concentration:** 90.4% of scraped companies with ≥5 SWE postings have ZERO entry rows under combined column; 49.3% under YOE proxy (T16).
- **AI Engineer title evolution:** 20 → 321 postings (16×); 0% → 45% agentic; 40% → 32% PyTorch; 15% → 54% LLM (V2 C).
- **LLM-authorship hypothesis:** REJECTED for length (+38% Q1), AI rate (+197% Q1), breadth; ATTENUATED for credential stacking (+29% Q1 vs +149% full) — report with the credential-stack caveat.

All numbers above have been cross-verified by V1 or V2. Numbers from individual task reports (T01-T29) that were NOT verified should be used with caution and re-checked before citing.
