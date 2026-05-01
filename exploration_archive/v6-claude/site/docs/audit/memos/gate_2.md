# Gate 2 Research Memo

Date: 2026-04-15
Wave: 2 (Open Structural Discovery, T08-T15)
Reports read: T08, T09, T10, T11, T12, T13, T14, T15
Per-orchestrator: this is the **most important gate** of the exploration.

---

## Gate 1 correction (must do first)

Gate 1 stated that "arshkon native `entry` rows have mean YOE 4.18 and 26% YOE ≥ 5." **T08 re-verified the number: mean YOE 4.18 is correct, but the YOE ≥ 5 share is 42.6%, not 26%.** Only 28.6% of arshkon native='entry' rows have YOE ≤ 2. The 2024 baseline is materially noisier than I wrote at Gate 1. The correction strengthens — does not weaken — Gate 1's "do not use `seniority_native` as a 2024 baseline" rule.

---

## What we learned

1. **The SWE posting market's dominant organizing principle is tech domain, not seniority.** T09's NMI diagnostic is decisive: cluster × tech_domain_proxy = 0.412, cluster × company_industry = 0.179, cluster × period = 0.027, **cluster × seniority_3level = 0.015**. Tech domain is 27× the seniority signal and 15× the period signal. The 20-topic BERTopic solution produces interpretable domains (LLM/GenAI, Defense/cleared, DevOps/SRE, JS frontend, Java enterprise, Data engineering, Embedded, .NET, Python backend, iOS, Android, QA, GPU/CUDA, plus 4 employer-template artifact topics). Asaniczka structured skills cross-validate at Spearman ρ = 0.947.

2. **Between 2024 and 2026, the AI tool and framework layer exploded as a first-class co-occurrence community.** T14's tech ecosystem mapping shows tech-network **modularity ROSE** (Louvain 0.56 → 0.66, 12 → 15 communities) and two **new** communities emerged: (a) an LLM/RAG applications cluster of 17 techs (langchain, langgraph, rag, openai_api, claude_api, agents_framework, mcp, fine_tuning, embedding, pytorch, scikit_learn, pandas, numpy, ml, nlp, llm, tensorflow), and (b) an AI-tools triad (copilot, cursor_tool, claude_tool) that did not exist as a connected subgraph in 2024. The AI explosion is not a vocabulary sprinkle — it is structural. This is the single cleanest empirical finding in the wave.

3. **Per-tool SNR from the calibration extension:**

   | Tech / construct | 2024 → 2026 | Δpp | SNR |
   |---|---|---|---|
   | claude_tool | 0.01% → 3.37% | +3.35 | **326** |
   | agents_framework* | 0.61% → 12.70% | +12.08 | 140 (*regex FP ~30%, true ~+10-11 pp) |
   | embedding | 0.15% → 2.82% | +2.67 | 123 |
   | copilot | 0.06% → 3.77% | +3.71 | 44 |
   | langchain | 0.11% → 3.10% | +2.99 | 36 |
   | AI-any | 5.15% → 28.63% | +23.5 | **925** (Gate 1, reconfirmed) |

   These SNRs are 1-2 orders of magnitude above the next-strongest robust findings. RQ3 is the lead RQ.

4. **Length growth is real content, not boilerplate — but almost none of it landed in the requirements section.** T13's section anatomy decomposition on LLM-cleaned text: `responsibilities` +196 chars (52% of growth), `role_summary` +139 chars (37%), `preferred` (nice-to-have) +111 chars (29%), **`requirements` −2 chars (flat)**, benefits + about_company + legal combined <1% of growth. This is unexpected. The naive "scope inflation" story had predicted growth in requirements; the actual growth is in responsibilities and role-summary prose. T12's section-filtered Fightin' Words cross-validates: 84 of 100 2026-heavy terms appear in both full-text and requirements+responsibilities-filtered top-100 lists, so the content signal is not boilerplate, but it IS mostly in responsibilities — not where a traditional "hiring bar rising" narrative would put it.

5. **T11's credential_stack_depth and requirement_breadth rise is robust and survives SNR, but roughly half of it is length-driven.**
   - `requirement_breadth` SNR 10.2 (6.88 → 9.27)
   - `credential_stack_depth_strict` SNR 10.5 (2.77 → 3.31)
   - `tech_count` SNR 5.4 (4.74 → 6.09), but `tech_density` per 1K chars is **FLAT, SNR 0.7** — all of the tech-count rise is length dilution
   - `scope_density` per 1K chars +85% (keyword-validation risk; do not treat as final until Wave 3 T22 validates)
   - Entry-level credential_stack_depth rose **+0.46 (seniority_final) / +0.61 (YOE ≤ 2)**, both operationalizations agree, breadth-excluding-AI rose +20-25% at entry. This is the clean version of the junior-scope claim that survives: entry postings are asking for more requirement *categories* simultaneously, including beyond AI vocabulary, partly due to length and partly due to genuine category stacking.

6. **The junior-share "rise" is an instrument + composition artifact — not a real increase.** Multiple independent lines of evidence converge:
   - **T08:** Under arshkon-only (cleaner 2024 baseline), `seniority_final` entry share of known FLIPS direction: 7.72% → 6.70% (−1.0 pp). Excluding TikTok/ByteDance: −1.7 pp. Excluding 7 entry-specialist intermediaries (SynergisticIT, WayUp, Jobs via Dice, Lensa, Emonics, Leidos, IBM — none flagged `is_aggregator`): −2.1 pp.
   - **T08:** Seniority-known denominator drifted 61% → 47% between periods because Stage-10 LLM budget exhausted on scraped. This alone creates denominator bias for any "of known" comparison.
   - **T08:** `seniority_final` entry share and YOE ≤ 2 entry share have <10% row overlap — they measure different populations. Capping cannot fix the specification dependence; it is structural.
   - **T09:** Within-archetype junior share rose **uniformly** across EVERY archetype (LLM 3.7→8.5, DevOps 1.6→5.0, JS 2.7→9.8, Java 3.7→4.1, Data 3.3→4.2, Embedded 2.4→7.7, .NET 2.6→8.9). Uniform within-group movement is a signature of measurement change, not composition or restructuring.
   - **T11:** `seniority_final`-based strict management rate 8.5% → 9.5% is within instrument noise (SNR 0.35).
   - **T12:** The relabeling diagnostic says entry 2026 postings contain *classic entry markers* (pursuing, currently, gain, coursework, internships, graduation, recommendation) PLUS AI-era content. They are not relabeled mid-senior postings. So the **content** of entry-labeled rows is real entry content; the **share** of them is instrument-driven. These are compatible.
   - Combined with the cooling JOLTS backdrop (info sector openings −29%), the cleanest reading is: absolute entry volume did not expand; the share appears to rise because (a) the 2024 baseline was under-labeled, (b) hidden intermediary specialists post cheap entry roles in bulk in 2026, and (c) the denominator drifted.

7. **Seniority levels are not converging — if anything, slightly diverging, and the divergence fails SNR.** T15's centroid similarity analysis: junior↔senior embedding cosine 0.981 → 0.959 (shift −0.022, SNR 1.90), TF-IDF 0.944 → 0.876 (shift −0.068, SNR 1.94). Both representations agree on direction (diverging) and both fail SNR ≥ 2 by a hair. Per-archetype follow-up: **no archetype converges.** The only archetype with clean within-2024 calibration (Java enterprise) is diverging with SNR 2.30. LLM/GenAI shows the largest divergence. **TF-IDF nearest-neighbor analysis: 2026 juniors still strongly resemble 2024 juniors (+12.1 pp excess over base rate, 2.4× the base).** The "seniority is blurring" thread of RQ1 is contradicted by both embedding and TF-IDF representations.

8. **2026 postings are internally MORE homogeneous than 2024**, not less. Average within-group pairwise cosine rose +0.017 (junior), +0.027 (senior), +0.040 (unknown). Role blurring would predict the opposite. Combined with the modularity rise (T14), the honest reading is that 2026 postings are being drafted to a more uniform template — consistent with recruiter tooling adoption (T29's LLM-authorship hypothesis is a candidate mechanism for Wave 3). This is template homogenization, not role restructuring.

9. **The "senior archetype" story is partially supported but changes shape.** T11 validated the management pattern: the broad pattern (precision 26%, half triggered by bare "team") fails validation; the strict pattern (manage/mentor/coach/hire/headcount) fails SNR. The **defensible** narrower finding is: `mentor` mentions rose 3.8% → 5.9% (+55%), while `manage` mentions FELL 3.8% → 2.0%. This is a mentoring-vocabulary shift, not a general management shift, and it is a cleaner and narrower version of RQ1's senior redefinition claim than the original. **New finding from T10:** at the title level, "senior" dropped 41.7% → 28.9% (−12.8 pp), "staff" rose +3.1 pp, "principal" and "architect" each fell. This is title-level senior-band compression, unanticipated by RQ1-RQ4 and the kind of diagnostic T21 should make its primary metric.

10. **Domain structure is real, but the original Alternative B composition mechanism is contradicted.** T09 established that archetype structure is by tech domain (confirmed), ML/AI grew +15.6 pp (confirmed), but ML/AI is NOT entry-poor (its junior share is comparable to other archetypes) and frontend did NOT shrink (+1.4 pp). The biggest shrinker was Defense/cleared (−14.8 pp), partly an arshkon composition artifact. The aggregate junior-share movement **cannot be decomposed to cross-domain composition** because within-domain movements are uniform. Alt B's "ML/AI eats frontend" mechanism is wrong, even though the domain dominance and AI growth facts underneath it are right.

---

## What surprised us

1. **Length growth landed in responsibilities and role_summary, not in requirements.** The requirements section is flat. This is unexpected — it disrupts the simple "hiring bar rising" reading of the length growth and complicates the scope-inflation story. It is more consistent with a "employers are describing the work in more detail and selling the role harder" reading than "employers are asking for more."

2. **Tech network modularity ROSE.** Prior theory suggested AI would blur technology bundles (e.g., ML becomes a generic skill attached to every role); the data shows the opposite. AI carved out new first-class communities with crisper boundaries, and existing communities consolidated. The coherent interpretation is that AI is a genuinely new specialty layer being stacked on top of existing stacks, not a diffused horizontal skill.

3. **2026 postings are more internally homogeneous than 2024.** Within-group pairwise cosine rose across all seniority levels. Against the intuitive "role-blurring" prior, postings are converging on templates. This is direct evidence for a recruiter-tooling / LLM-authorship mechanism (T29 in Wave 3).

4. **Senior title compression at the lexical level.** "Senior" in raw titles dropped 12.8 pp while "staff" rose 3.1 pp. Nobody had this on the board. T21 should now make this a primary diagnostic.

5. **Broad management patterns are half-driven by the single word "team."** 51% of broad-only hits triggered on bare "team", 19% on "coordinate", 11% on "lead" (as a verb), 5% on "leadership". Any published cross-period comparison using `\bteam\b` as management language is measuring a boilerplate artifact.

6. **Entry-specialist intermediaries are invisible to `is_aggregator`.** SynergisticIT, WayUp, Jobs via Dice, Lensa, Emonics, Leidos, IBM — all 100% entry-post specialists, none flagged. Gate 1's "TikTok/ByteDance ~10%" understated the true intermediary-driven share of the 2026 entry pool. The `is_aggregator` flag needs a companion "entry specialist" or "staffing firm" tag.

7. **Disappearing terms are all HTML-stripping artifacts, not real vanished technologies.** 14 disappearing terms top the list and every single one is an arshkon concatenation artifact (`experienceexperience`, `developerlocation`, `skillsability`). **No technology genuinely disappeared above threshold.** The 2024 → 2026 vocabulary change is purely additive — AI stack added on top.

8. **Tech density (per 1K chars) is flat for all SWE but shows a clear split:** AI-mentioning postings have density rising (+0.60) while non-AI postings have density FALLING (1.84 → 1.60). The length growth in non-AI postings is genuinely boilerplate-dilutive; in AI postings it is content-dense addition.

9. **`c++`, `c#`, `.net` are backslash-escaped in scraped markdown** — T14 flagged that the shared tech matrix may under-count them in 2026 rows. Any "legacy-language decline" finding is potentially an extraction bug. Recommended fix: `re.sub(r"\\([+\-#.&_()\[\]\{\}!*])", r"\1", text)` in the cleaned-text pipeline before the next rebuild.

10. **Arshkon native `entry` is even noisier than Gate 1 reported.** 42.6% of arshkon native='entry' is YOE ≥ 5 (I wrote 26% at Gate 1). Only 28.6% is YOE ≤ 2. The 2024 native-label pool is majority non-entry by any reasonable YOE definition.

---

## Evidence assessment

| Finding | Sample | Evidence strength | Calibration survives? | Sensitivity survives? |
|---|---|---|---|---|
| AI tool/framework explosion (aggregate +23.5 pp, 5.56×) | Full SWE | **Strong** | Yes (SNR 925, individual tools SNR 44-326) | Yes (aggregator, text source, source restriction, within-2024) |
| Tech network modularity rise (0.56 → 0.66) | Capped corpus | **Strong** | Yes (structural, not aggregate) | Consistent under aggregator exclusion |
| Two new tech communities (LLM/RAG, AI-tools triad) | Capped corpus | **Strong** | N/A (structural) | — |
| Domain dominance of clustering (NMI 0.412 vs 0.015) | Capped sample | **Strong** | — | Stable across seeds (large topics) |
| ML/AI archetype growth (+15.6 pp) | Full corpus | **Strong** | — | Driven partly by TikTok/ByteDance intern pipelines |
| Requirement breadth rise (6.88 → 9.27, SNR 10.2) | Full SWE | **Moderate** | Yes (SNR 10.2) | Half length-driven, half genuine after T13 partial-section accounting |
| Credential stack depth rise (strict) | Full SWE + entry subset | **Moderate** | Yes (SNR 10.5) | Robust across operationalizations; entry pattern holds |
| Entry-level credential stacking (+0.46 / +0.61 entry) | Entry subset (thin) | **Moderate** | Yes | Agrees under `seniority_final` AND YOE ≤ 2 |
| Length grew in responsibilities/role_summary, not requirements | Full SWE | **Strong** | N/A (structural) | Aggregator exclusion identical; raw-text sanity agrees |
| 84/100 overlap of top Fightin' Words terms (full vs section-filtered) | Capped corpus | **Strong** | — | — |
| 2026 vocabulary additive (no real disappearances) | Full SWE | **Strong** | — | — |
| Mentor rose +55% / manage fell −47% | Full SWE | **Moderate** | Sensible in light of SNR noise on broader management | Narrow, requires wave 3 pattern validation |
| Title-level senior compression (−12.8 pp) | Full SWE | **Moderate** | — | Not yet tested in Wave 3 |
| 2026 within-group homogenization | T15 sample | **Moderate** | — | Consistent across embedding + TF-IDF |
| Aggregate junior share rose (of known) | Full SWE | **Weak + artifact-likely** | **No** (SNR 0.17-0.23) | Flips under arshkon-only baseline + exclude intermediaries |
| Junior share fell within overlap panel (`seniority_final`) | 115 companies | **Moderate** | N/A | Specification-dependent vs YOE proxy |
| No seniority convergence (slight divergence) | T15 sample | **Moderate** | Fails SNR by a hair (1.90-1.94) | Consistent across embedding + TF-IDF + per-archetype |
| ML/AI is entry-poor (Alt B premise) | T09 per-archetype | **Contradicted** | — | — |
| General management rise | T11 validated sample | **Contradicted** | Strict fails SNR, broad fails precision | — |

**Summary:** the paper has one **rock-solid** empirical headline (AI tool/framework explosion), two **strong** structural results (domain dominance of clustering + length grew in responsibilities not requirements), three **moderate** content results (credential stacking, mentor/manage swap, senior title compression), and multiple **contradicted or artifact** results (junior share rise as headline, seniority convergence, Alt B composition mechanism, general management shift).

---

## Narrative evaluation

**Original RQ1-RQ4 framing:**

- **RQ1 junior scope inflation:** Contradicted as originally framed. The aggregate junior-share rise is instrument-driven. The scope-inflation component survives in narrow form — entry-level requirement_breadth and credential_stack_depth rose robustly (SNR 10.2-10.5) and the rise survives excluding AI vocabulary, but half is length-driven and the length grew in responsibilities/role_summary, not requirements. **Status: contradicted at aggregate, narrow survival at entry-level content.**
- **RQ1 senior archetype shift:** Partially supported in a narrower form than hypothesized. General management language fails validation (precision) or SNR (strict). The defensible claim is a **mentoring-vocabulary shift** (mentor +55%, manage −47%) plus a **senior title compression** (senior −12.8 pp, staff +3.1 pp, principal + architect declining). **Status: reframed as narrow lexical + title-level shift, not a full archetype redefinition.**
- **RQ2 requirement migration between seniority levels:** Partially supported. Entry-level credential stack depth rose while mid-senior content got longer/denser. No convergence between levels. **Status: partial support for downward migration of requirement category breadth, but not for level convergence.**
- **RQ3 employer/worker AI divergence:** Strongly supported. The AI requirement explosion is the cleanest signal in the dataset. Every sensitivity check passes. The decomposition (per-tool SNRs 44-326) is publication-ready. Wave 3 T23 still needs to do the worker-side benchmark comparison. **Status: elevated to lead.**
- **RQ4 mechanisms:** Out of scope for exploration, but the T15 homogenization finding + T29's planned LLM-authorship test in Wave 3 are the strongest exploration-side mechanism leads.

### Alternative narratives — where they stand after Wave 2

**Alternative A (instrument-dominated apparent junior-share change):** **Confirmed for the junior-share question.** T08 (arshkon-only flip, intermediary concentration), T09 (uniform within-archetype rise), T11 (management fails SNR), and the denominator drift collectively support it. But **not confirmed for the content question** — T12's relabeling diagnostic shows entry-2026 content is genuinely entry-level (pursuing, coursework, internships), just with AI vocabulary added. So the content of entry postings is real; the count/share is noisy.

**Alternative B (domain composition explains junior-share change):** **Structurally confirmed, mechanistically contradicted.** The domain structure is real (NMI 0.412); ML/AI did grow (+15.6 pp); but within-archetype entry share rose uniformly and ML/AI is NOT entry-poor. The "composition eats restructuring" mechanism does not hold.

**Alternative C (new, emerging from Wave 2) — "Tool-stack restructuring plus template homogenization":** The strongest single empirical signal is the AI tool/framework explosion, which restructured the tech co-occurrence graph (modularity 0.56 → 0.66, two new communities). Concurrently, 2026 postings are **more internally homogeneous** than 2024 under both embedding and TF-IDF. This is consistent with recruiter adoption of LLM drafting tools producing uniform templates. The paper under this framing is "the employer-side tool stack restructured, and the description text homogenized — both are genuine, but the seniority restructuring story is contradicted."

**My current ranking (weakest to strongest support):**
1. Original RQ1 junior narrowing (as written) — contradicted
2. Alt B cross-domain composition mechanism — contradicted
3. Entry-level credential stacking narrow RQ1b — moderate, survives SNR, half length-driven
4. Senior narrow redefinition (mentor up / manage down / title compression) — moderate, new diagnostics needed
5. Requirement migration narrow RQ2 — moderate, partial
6. Alt A instrument-dominated junior-share — strong for share, not for content
7. **Alt C tool-stack restructuring + template homogenization** — strong, multi-source cross-validated
8. **RQ3 AI requirement explosion** — strong, SNR 44-326, publication-ready headline

---

## Emerging narrative (one-sentence)

*"Between 2024 and 2026, software engineering job postings show a structural AI tool and framework explosion that carved new first-class technology communities and drove requirement breadth upward at every seniority level, while the aggregate junior-share 'decline' dissolves under instrument-calibration and the within-company evidence points to template-driven homogenization rather than role redefinition."*

That sentence is publishable. Wave 3 should either tighten it or complicate it — specifically by (a) validating or rejecting the T29 LLM-authorship mechanism, (b) testing whether the AI explosion is SWE-specific via T18 cross-occupation comparison, (c) computing the employer-usage divergence in T23, and (d) running the company typology in T16 on the overlap panel.

---

## Research question evolution

**Proposed updated RQ set for Wave 3 and analysis:**

- **RQ1 (revised):** How did SWE job posting requirement breadth, technology stack, and language change between 2024 and 2026, and how much of the apparent change is real content vs. length dilution vs. templated homogenization? (Replaces the original junior-narrowing framing with a content-change framing.)
- **RQ2 (revised):** Which specific AI tools, frameworks, and technology bundles drove the 2026 restructuring of the SWE technology co-occurrence graph, and how did domain archetypes (Frontend, Data, LLM/GenAI, Embedded, …) shift relative to each other? (Replaces the seniority-migration framing with a tech-stack + archetype framing.)
- **RQ3 (elevated, still named):** Do employer-side AI requirements outpace worker-side AI usage benchmarks? Wave 3 T23 must produce the divergence number; given T14's SNR 44-326 on per-tool prevalence, this is expected to be the paper's lead empirical claim.
- **RQ4 (kept, interview design unchanged):** How do senior engineers, junior engineers, and hiring-side actors explain the restructuring they see in the data? The T29 LLM-authorship finding (if positive in Wave 3) would become a mandatory interview probe — "did you write this posting, or did a tool draft it?"
- **RQ5 (new from Gate 1, now confirmed as worth a section):** Specification dependence of cross-period junior metrics is itself a methodological contribution. `seniority_final` ↔ YOE-proxy have <10% row overlap and yield different directions on the overlap panel. Labor-market posting research should not use a single seniority operationalization.
- **RQ6 (new, from T08):** The 2026 SWE entry pool is driven by a small number of specialist employer types — staffing firms (SynergisticIT, Emonics), college-jobsite intermediaries (WayUp), tech giants' intern pipelines (TikTok/ByteDance), and bulk-posting consulting (Leidos, IBM). The `is_aggregator` flag misses all of these. Entry-level analysis of posting data without accounting for this concentration produces false trend claims.

**Dropped:**
- The "ML/AI is entry-poor" alternative mechanism (contradicted by T09).
- The "seniority convergence" claim (contradicted by T15 under both embeddings and TF-IDF).

---

## Gaps and weaknesses

1. **Two out of ten archetypes are employer-template artifacts** (Amazon AWS boilerplate, Amazon program boilerplate, generic JS template, generic Python template). These appear as valid BERTopic topics because repeated postings cluster. T09 flagged them but future topic models should cap per-company more aggressively or deduplicate on `description_hash`.

2. **The `agents_framework` regex has ~30% false positives** in 2024 (call-center/change/biomedical agents). T14 documented this; the true AI-agent rise is ~+10-11 pp, not +12.08. Wave 3 T22/T23 should use a strict AI-agent pattern.

3. **The scraped text has backslash-escaped tech tokens (`c\+\+`, `c\#`, `\.net`).** The shared tech matrix may under-count these in 2026, producing false "legacy language decline" signals. Preprocessing owner should apply `re.sub(r"\\([+\-#.&_()\[\]\{\}!*])", r"\1", text)` before the next shared rebuild. Not a blocker, but Wave 3 agents should be warned.

4. **T09's archetype labels cover only 53.5% of SWE rows** (arshkon 99.9%, asaniczka 94%, scraped 30.7%). This is the `text_source = 'llm'` constraint. T28 and Wave 3 stratification by archetype will be coverage-limited on the 2026 side.

5. **The scope_density +85% finding (T11) has a keyword-validation risk similar to the management_broad issue.** T22 in Wave 3 should validate the scope patterns with the same 50-row stratified sample discipline that T11 used for management.

6. **The senior title compression is a correlational finding.** Why did "senior" drop and "staff" rise? Alternatives: (a) real senior-band inflation (staff is the new senior), (b) LinkedIn title-normalization drift, (c) employer template adoption of "staff" ladder. T21 should distinguish.

7. **No Wave 2 analysis has run on the overlap panel directly.** T16 does this in Wave 3. Until then, all "within-company" claims reference only T06's Wave 1 overlap panel findings.

8. **Entry-level sample remains thin for text-dependent analysis.** T12's secondary entry-2024 vs entry-2026 comparison was marginal; the cleaner relabeling diagnostic (entry 2026 vs mid-senior 2024) is the main entry-specific text finding.

---

## Direction for next wave

**Dispatch the V1 verification agent first**, before Wave 3. V1 should re-derive the top 5 Wave 2 numbers independently, validate scope_density patterns (which are flagged as high-risk), and test Alt A vs Alt C on at least one headline under alternative control definitions.

**Wave 3 dispatch with the following modifications to task prompts:**

- **Agent J (T16 + T17):** The company typology (T16) is now critical — it is the only remaining test of within-company behavior. Explicitly look for a "tool-stack adopters" cluster among the overlap-panel companies (did some companies pivot harder toward AI vocabulary than others?). Split entry-share decomposition findings by **archetype** using T09 labels. For T17, validate whether metro-level AI surge correlates with metro-level entry-share movement — a positive correlation is a testable implication of Alt C.
- **Agent K (T18 + T19):** The cross-occupation SWE-vs-adjacent-vs-control DiD is critical for the paper's SWE-specificity claim. If the AI explosion appears in all three groups, our story changes from "SWE restructured" to "information work restructured." For T19, verify that the JOLTS-driven macro cooling isn't confounding our cross-period effects by computing the cross-period effects within a single source window rather than pooled.
- **Agent L (T20 + T21):** The senior title compression is new — T21 should make it the primary diagnostic. Validate the mentor/manage swap with a 50-row stratified precision check (same method as T11). For T20, the boundary-blurring question now has a Wave 2 answer (NO — levels diverged slightly). T20 should pivot to **explaining** the divergence: what features separate entry from mid-senior more cleanly in 2026 than in 2024?
- **Agent M (T22 + T23):** T22 must validate the scope_density +85% finding with 50-row stratified precision checks. T23 is now the highest-stakes task in Wave 3 because it produces the RQ3 headline divergence number. Use the T14 per-tool SNRs as the employer-side inputs. For the worker-side benchmark, try Anthropic's occupation usage data first, StackOverflow second.
- **Agent O (T28 + T29):** T28 runs the deferred T08-step-7 and T11-step-7 archetype decomposition and will directly test whether the uniform within-archetype junior-share rise holds under `seniority_final` AND YOE ≤ 2. T29's LLM-authorship hypothesis is now the **emerging narrative's mechanism candidate** — if supported, it becomes a core contribution; if rejected, the "template homogenization" reading has to find a different mechanism. Treat T29 as promoted from exploratory to primary.

**Tell every Wave 3 agent:** the junior-narrowing story is contradicted. The lead finding is the AI tool/framework explosion. The emerging narrative is Alt C (tool-stack restructuring + template homogenization). Wave 3 should either tighten Alt C or replace it with a better-supported alternative.

---

## Current paper positioning

**If we stopped here, the paper is:**

### Lead claim
"Between 2024 and 2026, software engineering job postings underwent a structural restructuring of their technology stack — a new AI tool and framework community emerged as a first-class co-occurrence cluster, and aggregate AI requirement prevalence rose 5-fold (5.15% → 28.63%) against a cooling labor market. The traditional 'junior rung narrowing' story does not survive specification-dependence and instrument-calibration checks; the real employer-side change is in what technologies employers list, not in how they describe seniority levels."

### Contributions (ranked)

1. **A longitudinal SWE postings dataset** with transparent preprocessing, three independent sources, LLM-budget-aware frame. BLS r=0.97 geographic validation.
2. **A rigorous robustness framework for posting-based labor research.** Within-source calibration, specification dependence diagnostics, concentration prediction, archetype structure. The junior-share null result under this framework is itself a methodological contribution — "single-operationalization posting analyses produce false trends."
3. **The AI tool/framework explosion quantified at per-tool granularity** — claude_tool SNR 326, copilot SNR 44, langchain SNR 36, tech network modularity 0.56 → 0.66, two new co-occurrence communities identified. This is the paper's empirical headline.
4. **The entry-level credential-stacking narrow claim** — entry postings ask for more requirement categories simultaneously (+0.46 to +0.61 credential stack depth at entry), half length-driven, half real content, under both seniority operationalizations. This is the surviving sliver of RQ1.
5. **The senior archetype shift as narrow lexical change** — mentor +55%, manage −47%, senior title compression 41.7% → 28.9%. This is the surviving sliver of the senior-redefinition claim.
6. **Template homogenization (2026 postings more internally uniform than 2024)** — points toward recruiter LLM-drafting adoption as a candidate mechanism for Wave 3 T29.

### What Wave 3 needs to deliver to strengthen it

- **T23 divergence number** for the RQ3 lead (strongest expected contribution to the headline).
- **T18 cross-occupation DiD verdict** (SWE-specific or field-wide?).
- **T16 company typology** within the overlap panel (is there a "tool-stack adopter" cluster?).
- **T28 archetype decomposition** under both entry operationalizations (does the uniform within-archetype rise hold on the better-defined T09 labels?).
- **T29 LLM-authorship verdict** (is the homogenization story mechanized by recruiter tooling?).
- **T21 senior-band diagnostic** (does the title-compression finding survive and what drives it?).

### What we should NOT claim (updated)

- "Junior share declined" — contradicted.
- "Seniority levels converged" — contradicted.
- "ML/AI eats frontend" — contradicted.
- "Management language rose" — contradicted.
- "Roles are blurring" — contradicted (opposite direction found under homogenization reading).

---

## Decisions going into V1 verification and Wave 3

1. **Lead RQ: AI requirement explosion (RQ3).** Elevate throughout.
2. **Secondary RQs: narrow scope inflation (entry credential stacking) + narrow senior shift (mentor/manage + title compression).**
3. **Dropped claims: junior share decline, seniority convergence, general management rise, ML/AI-eats-frontend composition mechanism.**
4. **New RQs: RQ5 (specification dependence as methods contribution), RQ6 (entry-poster concentration).**
5. **Mandatory wave 3 validations: scope_density patterns (T22), AI agent regex cleanup (T22), LLM-authorship hypothesis (T29 promoted).**
6. **Preprocessing owner action: apply `re.sub(r"\\([+\-#.&_()\[\]\{\}!*])", r"\1", text)` fix to cleaned text pipeline; tag entry-specialist intermediaries with a new flag separate from `is_aggregator`.**
7. **Gate 2 clears, conditional on V1 verification of (a) per-tool SNRs in the calibration table, (b) section-anatomy decomposition in T13, (c) within-archetype uniform entry rise in T09, and (d) management pattern precision claims in T11.**
