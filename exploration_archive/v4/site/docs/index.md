# Exploration Task Index

Last updated: 2026-04-10 (Gate 1 complete + seniority deep-dive)

## Seniority recommendation (Gate 1, revised)

**Primary: the combined best-available column.** Stage 10 routes rows: high-confidence rule-based rows are skipped (`llm_classification_coverage = 'rule_sufficient'`, `seniority_llm` NULL — read `seniority_final`), and low-confidence rows are sent to the LLM (`labeled`, `seniority_llm` populated). Construct:

```sql
CASE
  WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
  WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
  ELSE NULL
END AS seniority_best_available
```

Outside the core frame, fall back to `seniority_final`. See `docs/preprocessing-schema.md` "Recommended seniority usage."

**Required co-equal validator: a label-independent YOE-based proxy.** Compute the share of postings with `yoe_extracted <= 2` (and `<= 3`) by period. This does not depend on any seniority classifier. **If label-based and YOE-based findings disagree on the direction of an entry-level trend, do NOT pick a side without investigating WHY** — see the seniority measurement investigation report for the methodology.

**Critical Wave 1 + V1 finding:** Native-label-based entry-share comparisons are biased by differential label quality across data snapshots. 41% of arshkon's `seniority_native = 'entry'` SWE rows have `yoe_extracted >= 5` (vs 9.5% in scraped). The "entry decline" under `seniority_native`/`seniority_final` is largely a measurement artifact. Under YOE-based and combined-column measures, **entry share is rising** (V1 verified: combined +3.48pp, imputed +4.66pp, YOE proxy +6.27pp). The rise is robust to dedup, aggregator exclusion, top-20 removal, and company capping (V1 could not reproduce T08's reversal except in one narrow noise-level cell). The 2026 entry-share rise is primarily driven by genuine large-employer new-grad programs (Google, Walmart, Qualcomm, SpaceX, Amazon, Microsoft, Meta), with ~23% of the small combined-column entry pool contaminated by exact `description_hash` duplicates from 6 companies (Affirm, Canonical, Epic, Google, SkillStorm, Uber) — a preprocessing dedup issue, not a data interpretation problem. See `exploration/reports/seniority_measurement_investigation.md` and `exploration/reports/V1_verification.md`.

**Asaniczka rule:** Asaniczka has zero native entry labels.
- Under `seniority_native`/`seniority_final`: do NOT pool asaniczka into 2024; use arshkon-only baseline.
- Under combined best-available column, `seniority_imputed`, and YOE proxy: asaniczka can be included.

**Ablation set to report on every seniority-stratified finding:**
1. Combined best-available (primary)
2. `seniority_native` (arshkon-only baseline for entry analyses)
3. `seniority_final` (arshkon-only baseline for entry analyses)
4. `seniority_imputed` (where != unknown)
5. YOE-based proxy (`yoe_extracted <= 2` share)

## Wave 2 findings at risk after V1 verification (Wave 3 agents read this)

The following Wave 2 claims need to be re-stated, qualified, or used with care:

- **T14 "5 declining technologies" list is wrong.** V1 found `swe_tech_matrix.parquet` has a broken regex for c_cpp (`\bc\+\+\b` cannot match) and csharp. Corrected mention rates: **C++ 15.6% → 19.0% (+21% growth, NOT a decliner)**, **C# 11.1% → 11.2% (flat, not declining)**. T14's network is missing a systems community anchor. Wave 3 tasks that depend on the tech matrix should treat c_cpp/csharp as suspect until the matrix is regenerated. The engineer handoff doc is `docs/preprocessing-dedup-issue.md`; the regex bug is separately documented in V1's report.
- **T13 "57% length growth is genuine content" needs the text-source caveat.** Under LLM-text-only subset, length growth is only +26% (vs +72% mixed). About half of the aggregate length growth reflects text-source composition (2024 91% LLM-cleaned vs 2026 21% LLM-cleaned), not real content expansion. Length growth IS still real, just smaller than headlined.
- **T12 uncapped Fightin' Words top-30 contains volume-driven artifacts** (`accommodation`, `usd`, `values`, `americans`, `marrying`, `experimenting` etc. are Amazon-driven boilerplate, not genuine cross-period shifts). The AI vocabulary subset of T12's top-30 is robust under capping; the credential-stripping subset (`qualifications`, `degree`, `bachelor`) is robust. Treat the non-AI / non-credential bigrams in T12 with care.
- **T11 / T08 `agent` indicator is contaminated.** ~25-45% of `agents?` matches are non-AI uses (insurance agents, legal disclosures, robotics autonomy, "change agent"). The +2180% `desc_contains_agent` figure from T08 is directionally correct but inflated. The `agentic` token alone is ~95% AI-precision and is the cleaner signal. Wave 3 AI-mention analyses should use `agentic` or `AI agent` / `multi-agent system`, not bare `agent`.
- **T06 within-company decomposition (legacy)** used contaminated native labels. T16 in Wave 3 is the corrected version under combined column + YOE proxy.
- **T08 "company-capping reverses entry rise"** finding could not be reproduced by V1 except in a narrow `arshkon-only × YOE × cap20` corner at -0.8pp (within noise). Under pooled-2024 baseline + dedup + cap20, the rise is robust. Treat the entry rise as **real, primarily from large-employer new-grad programs**, not as concentration-driven.

**Robustly verified by V1 (use with confidence):** AI/ML domain growth +11pp; credential stack depth 5.59×; entry-share rise under combined column / imputed / YOE proxy; AI mention quadrupling 13.7%→52.0% (with the `agent` caveat above); strict-mentoring growth at ~100% precision; mid-senior credential vocabulary stripping; T15 semantic-convergence null (holds under text-source control).

## Wave 1 — Data Foundation

| Task | Agent | Status | Key finding | Report |
|---|---|---|---|---|
| T01 | A | **done** | 63K SWE after filters. Scraped LLM text at 21% is #1 gap. Scraped now spans 2026-03 + 2026-04 (35K SWE). | [T01.md](reports/T01.md) |
| T02 | A | **done** | Asaniczka associate is NOT a junior proxy (YOE 4.0, 72% classified associate). 2024 entry baseline = arshkon-only 769 rows. | [T02.md](reports/T02.md) |
| T03 | B | **done** | **Entry-share direction flips by operationalization.** Native/final (arshkon): decline 22→14%. LLM: increase 17→28% (n too small). seniority_imputed: no change. Resolved by deep-dive (see below). | [T03.md](reports/T03.md) |
| T04 | B | **done** | SWE classifier solid: 91% regex tier, no dual-flags. title_lookup_llm has 22% FPR. QA/test roles main contamination. | [T04.md](reports/T04.md) |
| T05 | C | **done** | Description length +57% (likely real). Title vocabulary stable (Jaccard 0.43). YOE stable within matched cells. "AI Engineer" exploded 6→321. | [T05.md](reports/T05.md) |
| T06 | C | **done** ⚠ | Within-company entry decline 28.7%→12.4% under `seniority_native`/`seniority_final`. **Inputs are contaminated** — defer interpretation to T16 re-run with combined column + YOE proxy. | [T06.md](reports/T06.md) |
| T07 | D | **done** | Geographic r=0.985 vs BLS. All SWE comparisons well-powered. Entry seniority_final adequate (MDE 1.6pp). JOLTS: arshkon at hiring trough. | [T07.md](reports/T07.md) |
| Sen-DD | (deep-dive) | **done** | **Native-label entry decline is a measurement artifact.** 41% of arshkon native-entry rows have YOE>=5 (vs 9.5% scraped). Label-independent (YOE proxy) shows entry share stable or modestly increased. Employer labeling explicitness increased significantly (+10.6pp imputed-known rate). | [seniority_measurement_investigation.md](reports/seniority_measurement_investigation.md) |

## Wave 1.5 — Shared Preprocessing

| Task | Agent | Status | Key finding | Report |
|---|---|---|---|---|
| Prep | Prep | pending | — | — |

## Wave 2 — Open Structural Discovery

| Task | Agent | Status | Key finding | Report |
|---|---|---|---|---|
| T08 | E | **done** | Entry share rises under 3/4 operationalizations + YOE proxy. core_length +59% (d=0.85). AI vocab: agent +2180%, rag +5600%. ⚠ Company-cap-20 REVERSES entry rise — small set of high-volume cos drives it. | [T08.md](reports/T08.md) |
| T09 | F | **done** | **AI/ML archetype +10.96pp** (the only large grower, absorbs share from .NET, frontend, Java, data-eng). Market organizes by tech domain, NOT seniority (NMI domain×primary_language=0.11 vs ×seniority=0.015). AI/ML is structurally LESS junior-heavy. Saved `swe_archetype_labels.parquet`. | [T09.md](reports/T09.md) |
| T10 | G | **done** | Title space fragmenting (314→378 unique/1K). AI/ML in titles 9.8%→23.9%. **Within "software engineer" string, entry share 1.2%→15.3%.** Junior markers tripled. Legacy stack titles vanished. | [T10.md](reports/T10.md) |
| T11 | G | **done** | **Credential stack depth 7: 3.8%→20.5% (5.4×).** requirement_breadth +37%, scope +97%, ai_mention +296%, tech density FELL 26%. Entry scope inflation robust across all operationalizations. **Mgmt naive trigger 8.4%→34.5% but ~85% boilerplate; strict detector 9.6%→13.5%; people-mgr terms ALL DECLINED — story is mentoring not mgmt.** | [T11.md](reports/T11.md) |
| T12 | H | **done** | AI vocab dominates: agentic 446×, rag 76×, claude 350×, cursor 604×, langgraph 0→2%, mcp 29×. **Mid-senior 2026 LOSES qualifications/required/degree/bachelor**. Relabeling diagnostic PASSES (2026 entry ≠ 2024 senior relabeled). BERTopic AI/ML +11.95pp (corroborates T09). ⚠ Arshkon entry pool finance/banking-skewed; Amazon dominates uncapped bigrams. | [T12.md](reports/T12.md) |
| T13 | H | **done** | **57% length growth is GENUINE content (+88.9% in core sections, 85.7% of gross delta).** Responsibilities +85% > Requirements +60%. Readability improved (FK 18→16.7), inclusive +41%, imperative -18%. Entry-vs-mid-senior readability gap compressed 2→0.6 grades. Section classifier saved as shared artifact. | [T13.md](reports/T13.md) |
| T14 | I | **done** | **AI cluster fusion** — Copilot/Claude/HF/RAG fuse with PyTorch/TensorFlow into 29-member ML+GenAI mega-community. AI mention 2.8%→26.2% (9.4×). 36% denser co-occur graph. Tech count 5→8 (median, +60%) but density -25% (length artifact). ρ=0.77 vs structured. **🔴 CRITICAL BUG: c_cpp/csharp regex broken; matrix says 0.5%/0.15%, actual 19.2%/11.3%.** | [T14.md](reports/T14.md) |
| T15 | I | **done** | **Convergence hypothesis REJECTED.** Junior↔senior cosine flat (-0.001 emb, -0.022 TF-IDF). Below noise floor by 4×. NN: 2026 entry retrieves 2024 entry+senior, away from 2024 mid (mild polarization, not inflation). Dominant variation = period > seniority. ⚠ text_source confound: 2024 94% LLM-cleaned vs 2026 79% rule-cleaned. | [T15.md](reports/T15.md) |

## Wave 2 Verification

| Task | Agent | Status | Key finding | Report |
|---|---|---|---|---|
| V1 | V1 | **done** | All Part-A headlines re-derived within 5% (AI/ML +11-14pp, credential depth 5.59×, entry rise robust under combined col, length +72%). Strict mentoring precision ~100%; naive hire precision only 8-28% (T11 correct). **C++ corrected: +21% growth, NOT a decliner; C# flat.** Top 20 combined-col entry contributors are 23% duplicate-template artifacts (Affirm/Canonical/Epic/Google 14-25× dup); 2026 entry rise is robust to cap20, dedup, aggregator exclusion, and top-20 removal under both combined column and YOE proxy (**did NOT reproduce T08's reversal** except in the arshkon-only × YOE × cap20 corner at -0.8pp). T15 null holds under LLM-text-only subset (not a text-source confound). Capped FW: AI terms (workflows, observability, llm, rag, agent) survive capping; benefits/legal boilerplate in T12 uncapped top-30 is Amazon/volume-driven artifact. **Text-source-controlled length growth is only +26% vs aggregate +72% — much of the length story is text-source composition**. | [V1_verification.md](reports/V1_verification.md) |

## Wave 3 — Market Dynamics & Cross-cutting

| Task | Agent | Status | Key finding | Report |
|---|---|---|---|---|
| T16 | J | **done** | **90.4% of scraped cos with ≥5 SWE have ZERO entry rows (combined col); 49.3% under YOE proxy.** Top 20 YOE entry posters dominated by Google, Walmart, Qualcomm, SpaceX, Microsoft, Amazon, defense contractors. **Within/between decomposition: YOE entry rise is 87% between-company (+5.68pp), only 13% within-company (+0.82pp)** — 2026 reweights toward companies that were already entry-heavy, NOT employers pivoting to new-grad. AI rate is 91% within-company (real), scope 79% within (real), tech count 113% within. k=5 clusters: high-tech (keyword stuffing), low-tech, AI-forward+length, high-scope, entry-downsizer (n=18). **Combined-col within-company Δ entry is -0.27pp (slightly negative).** Native/final labels show -9.9pp decline (80% within), driven by contamination (per V1). New entrants slightly more AI/scope, slightly less entry than returners. | [T16.md](reports/T16.md) |
| T17 | J | **done** | 26 metros ≥50 SWE both periods. **AI↔scope metro correlation +0.43 (p=0.03)**, but **AI↔entry uncorrelated (r=-0.09, p=0.65)** — AI surge and entry rise are geographically decoupled. AI gains broad: Tampa +30pp, SLC +27pp, Miami/NYC/SF +26pp; Detroit/DC/LA/Denver lagging at +11-12pp. Entry (YOE) top: San Diego +26pp (Qualcomm effect), SF Bay / Seattle +12pp. **Hubs lead non-hubs modestly** (+3pp AI, +5pp scope). **Remote share 0%→22.3%** corpus-wide; remote and onsite have identical AI/scope rates. AI/ML archetype concentrated in SF (26.7%), Boston, Seattle; **Houston at rank 4 (15.6%) is a surprise**. NYC 26× coverage expansion flagged. | [T17.md](reports/T17.md) |
| T18 | K | **done** | **AI restructuring is tech-cluster, not SWE-specific.** SWE AI rate +22.9pp, adjacent +19.0pp, control +1.2pp. DiD SWE−control=+21.7pp (strong); DiD SWE−adjacent=+3.9pp (weak). SWE↔adjacent boundary slightly DIVERGED (cosine -0.022); **SWE↔control CONVERGED (+0.079)**. Length growth under LLM-text control is LARGER in adjacent than SWE (+629 vs +527 chars). Scope language Δ: adjacent +26.3pp > SWE +19.0pp (adjacent exceeds SWE). "AI Engineer" title evolved from pytorch/ML role (0% agentic, 22% pytorch) to LLM-agent role (38% agentic, 66% llm-stack). Cross-validation: tech-cluster restructuring at 83% of SWE magnitude in adjacent, 5% in control. **Weaken "SWE-specific" framing to "tech-cluster-specific."** | [T18.md](reports/T18.md) |
| T19 | K | **done** | Three-snapshot data, not time series. Cross-period AI acceleration ratio ~4× over within-2024 (agentic ~14×); scope ~2×; entry/tech count within-noise. 2026-03 vs 2026-04 stable on all content metrics except entry_best (-3.1pp, dedup artifact). Within-arshkon: entry_native 17.6%→14.5% across 2 weeks (non-trivial internal heterogeneity); arshkon is effectively a 3-day snapshot (56% posts on Apr 17-19). Day-of-week effect ~5pp on AI rate, ~4pp on entry, ~600 chars on length. `posting_age_days` unusable (1% coverage). First scrape day is NOT a backlog. **Mean tech count is nearly flat across snapshots (2.49→2.57→2.73) using a safe 15-tech list — contradicts T11/T14 tech-count growth claims; flag for V2.** | [T19.md](reports/T19.md) |
| T20 | L | **done** | Feature-based boundaries did NOT blur market-wide. **Only mid-senior↔director blurred** (AUC 0.75→0.69 under seniority_final) — people-mgmt dropped out of top discriminators, mentoring flipped to negative director predictor (mentoring drifted down from director to mid-senior). Entry↔associate was always fuzzy (AUC ≈0.55 both years). Associate tier shrank 41% and drifted toward mid-senior in feature space (centroid ratio 0.61→0.78). YOE proxy disagrees on early-career: yoe_low↔yoe_mid sharpened on non-YOE features (0.60→0.64), driven by edu becoming stronger negative discriminator. | [T20.md](reports/T20.md) |
| T21 | L | **done** | **T11 IC+mentoring reframing confirmed and sharpened.** Senior people-mgmt density −58%, mentoring density +33% (any-share 11%→22%), tech-orchestration +47% (any-share 37%→62%, biggest shift). New AI-orchestration vocabulary: `agentic` 415×, `prompt engineering` 20×, `guardrails` 15×. K-means: People-Manager cluster 3.7%→1.1% (−70% rel), Mentor-heavy 8.6%→10.8% (+26%), TechOrch-heavy 5.0%→7.6% (+52%). **Directors 2026 collapse to mid-senior profile** on people-mgmt/mentor/AI (explains T20 boundary blur); only strategic scope distinguishes directors. **2024 AI-mentioning seniors were mentor-heavy; 2026 are tech-orch-heavy** (profile flipped). **IC+mentoring NOT uniform across domains:** mentoring GREW at Frontend/Backend/Cloud, DECLINED at AI/ML (+tech-orch instead), Backend/Enterprise is only domain where people-mgmt rose. Senior rebranding is senior-specific (+0.003pp entry vs −0.016 senior on people-mgmt). Credential-stripping weakly correlates with AI mention (ρ=−0.15). | [T21.md](reports/T21.md) |
| T22 | M | **done** | **AI is ~10× more hedged than non-AI content within the same postings**: AI-term 80-char windows have hedge:firm ratio ~10:1 vs global ~1.5:1. AI-containing posts have hedge_any 66-74% vs non-AI 53-64% and firm_any 27-44% vs non-AI 38-46%. Aspiration-ratio>1 entry 23.7%→33.6%, mid-sen 16.4%→26.8%. Kitchen-sink≥12 tripled (entry 3.7%→12.6%, mid-sen 7.7%→27.3%). **Combined-column and YOE-proxy entry top-20 ghost lists have ZERO overlap** — combined surfaces real new-grads with dense stacks (IBM intern, GM intern, ByteDance grad); YOE surfaces senior roles mis-captured by "2+ yrs" phrase (Visa Staff SWE, CVS Senior, Xylem Senior AI). Template saturation rose 0.604→0.669 (mean pairwise cos) after exact-hash dedup; dedup halved the saturated-company share (11.9%→5.8%), confirming V1's 6-company dup artifact. **Direct employers are MORE ghost-like than aggregators** on kitchen-sink, any_ai, agentic — naive "aggregators are ghostier" assumption is wrong. yoe_scope_mismatch concentrated in native labels (25.5% entry→yoe≥5 at 2024); under combined column only 2.9%. Credential contradictions <0.1%, not a useful marker. Validated patterns saved to `exploration/artifacts/shared/validated_mgmt_patterns.json` (50-sample precision review ≥90% for all patterns; MCP removed from ai_tool due to Microsoft Certified Professional contamination). | [T22.md](reports/T22.md) |
| T23 | M | **done** | Employer AI requirement rate 14.3%→51.2% (+36.9pp, 3.6×); direct-only 11.2%→52.9% (+41.7pp, 4.7×). `ai_tool` subset (copilot/cursor/llm/langchain/prompt engineering) 1.6%→17.7% (**10.2×**, fastest-growing category). `ai_domain` 10.9%→22.8% (2.1×). `agentic` 0%→8.0%; `rag_phrase` 0.1%→5.6%. Entry combined 17%→43.6%, mid-senior 14.4%→43.4% — AI explosion broad-based across seniority. **Under all 4 benchmark scenarios employer requirements REMAIN BELOW worker usage in both periods** (direct-only 2026 any_ai 52.9% vs SO-central 80%; divergence -27pp). But employer side grew ~10× faster in relative terms (2.6-4.7×) than worker AI usage (1.29×, per SO 62%→80%). **Narrative correction: not "employers demand AI workers don't have" — it's "employers were slow to update JDs in 2024 (50pp lag) and are racing to catch up, still below the line but closing fast."** Agentic-specific divergence: 8% req vs 24% SO daily+weekly agent usage = -16pp gap; largest unclosed divergence. Growth divergence robust across all 4 benchmark scenarios. | [T23.md](reports/T23.md) |
| T28 | O | **done** | Scope inflation is UNIVERSAL within every archetype (+17-50% breadth, +4-107% tech, +41-153% scope, +1.5-64× credential stack ≥7). **Entry-share rise is 90-105% within-domain**; between-domain (AI/ML composition) contributes only +0.13pp. Senior-tier mentoring growth is the single cleanest cross-domain result: +1.9 to +29.4pp in every archetype; strict people-mgr language near 0 universally. **T09's "AI/ML is less junior-heavy" is a combined-col routing artifact**: under YOE proxy, AI/ML 2026 entry share 16.0% vs rest 17.2% (identical). AI/ML growth (+16.5pp within domain) is from existing large employers diffusing (Microsoft 84, JPMC 56, Amazon 52, Uber 40), not new entrants; top-20 concentration fell 31%→22%. Propagated T09 labels to 63K rows via nearest-centroid (87% holdout acc). | [T28.md](reports/T28.md) |
| T29 | O | **done** | **Hypothesis REJECTED.** Wave 2 findings survive (often strengthen) in low-LLM-score subset: length +77% low vs +79% high, credential stack ≥7 **+786% low** vs +384% high, AI mention +339% low vs +269% high, tech density -23% low vs -34% high, requirement breadth +42% low vs +35% high. **Cross-posting variance UNCHANGED** 2024→2026 (std 2.25→2.19), contradicting "LLMs uniformize" prediction. Em-dash density 9× and vocab density +36-63% DO rise under text-source control but are decoupled from headline content changes. Score is confounded with polished-marketing-prose (Microsoft 5.8→7.1 SD above mean in both years). Candidate LLM-adopters: Deloitte, PwC, American Express, Collins Aerospace. T13's length growth and T11's credential stacking are REAL CONTENT changes, not authoring-tool artifacts. | [T29.md](reports/T29.md) |

## Wave 3 Verification

| Task | Agent | Status | Key finding | Report |
|---|---|---|---|---|
| V2 | V2 | **done** | **Tech-count reconciled: ~16-27% mean growth (3→4 median), NOT 34-60%** — V1 c++/c# fix is not the only matrix issue; T11/T14 broken matrix, T19 too-narrow list. T16 87% between-company replicates EXACTLY under arshkon-only convention but **fragile** (0.5%-50% under alternative panels) — report with caveat. **T18 STRENGTHENED:** SWE-adjacent AI growth ~105% of SWE (30.4 vs 29.0pp), not 83%. T22 hedge ratio 10.6→11.6 verified; structural temporal stability holds. T23 RQ3 inversion verified (gap ~24-27pp employer < worker). **T29 mostly verified, ONE qualification:** credential stack +29% in Q1 vs +149% full corpus — partially attenuated; LLM-authorship rejection still holds for length/AI/breadth but credential-stack is partially writing-style mediated. **Cross-task pattern inconsistency: T22 strict_mentor 47-65% broader than T11; growth rates consistent. Wave 4 should standardize on T22 patterns.** | [V2_verification.md](reports/V2_verification.md) |

## Wave 4 — Integration & Synthesis

| Task | Agent | Status | Key finding | Report |
|---|---|---|---|---|
| T24 | N1 | **done** | 12 new hypotheses (6 highest-priority): H1 AI maturation research→production, H2 AI/ML early-career-prohibitive, H3 defense contractors drive between-company entry rise, H4 credential stripping ↔ AI mention growth at posting level, H5 RQ3 inversion mechanically caused by credential stripping, H10 AI-senior profile flip is compositional. 14 surprises catalogued. 5 key tensions for analysis phase: (1) credential-stack low-LLM partial mediation, (2) T16 87% convention dependence, (3) native-vs-YOE direction flip, (4) **unexplained content expansion mechanism** (LLM rejected, domain composition rejected — what's left?), (5) AI/ML mentoring exception mechanism. | [T24.md](reports/T24.md) |
| T25 | N2 | **done** | 7 interview artifacts (PNG + README) in `exploration/artifacts/interview/`: inflated junior JDs (anonymized), paired same-company JDs (ServiceNow/AT&T/Deloitte/Adobe), entry-share trend under 3 operationalizations annotated with LLM releases, senior 3-axis archetype shift, AI vocabulary emergence from zero (11 terms), posting-usage divergence (RQ3 inverted), defense contractor entry over-representation. | [README](../artifacts/interview/README.md) |
| T26 | N3 | **done** | **`SYNTHESIS.md` written** — 17 sections, ~9000 words, canonical handoff for analysis phase. Consolidates V1+V2 corrections into current state of play (NOT chronological). Replaces T11/T14/T19 tech-count headlines with V2's ~16-27% mean / 3→4 median. Reports 50-87% between-company entry decomposition with panel-convention caveat. Strengthens tech-cluster-wide framing (~100% adjacent vs SWE). Standardizes on T22 validated patterns. Includes canonical citations appendix with V1/V2 verification status per number. | [SYNTHESIS.md](reports/SYNTHESIS.md) |

## Wave 5 — Presentation

| Task | Agent | Status | Key finding | Report |
|---|---|---|---|---|
| T27 | P | pending | — | — |

## Data constraints (updated Gate 1)

- **Total SWE (LinkedIn, Eng, ok):** arshkon 5,019 | asaniczka 23,213 | scraped 35,062
- **LLM extraction coverage:** arshkon 99%, asaniczka 90%, scraped 21% ← BINDING CONSTRAINT
- **LLM classification:** arshkon 100% (labeled+rule_suff), asaniczka 34%, scraped 21%
- **seniority_llm:** ~85% unknown rate; 62% of native-entry classified as mid-senior; NOT usable as primary
- **Entry-level (seniority_native):** arshkon 769, asaniczka 0, scraped 3,972
- **Entry-level (seniority_final):** arshkon 848, asaniczka 129 (artifact), scraped 4,656
- **Entry-level (seniority_llm):** arshkon 84, asaniczka 39, scraped 180 — UNDERPOWERED
- **Company panel (>=3 SWE both periods):** 228 companies (arshkon-only); 541 (pooled 2024)
- **Metro feasibility:** 8 metros with >=100 arshkon SWE; all 26 with pooled 2024 >=50
- **JOLTS confound:** arshkon at hiring trough (87K), asaniczka at uptick (160K)
- **Geographic representativeness:** r=0.985 vs BLS OES
