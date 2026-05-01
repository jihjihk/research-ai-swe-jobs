# Gate 2 Research Memo

**Date:** 2026-04-10 (V1 verification addendum added 2026-04-10)
**Wave completed:** Wave 2 (T08-T15), Agents E-I, plus V1 verification
**Reports:** T08.md, T09.md, T10.md, T11.md, T12.md, T13.md, T14.md, T15.md, V1_verification.md

---

## V1 verification addendum (read first)

V1 was dispatched after the body of this memo was written. Its findings change several specific claims while strengthening the overall narrative. Read this addendum first, then read the rest of the memo with these adjustments in mind.

### What V1 confirmed

- **AI/ML domain restructuring (T09 +10.96pp / T12 +11.95pp):** Verified independently at +11.86pp via title-based proxy and +13.69pp via strong-keyword body proxy. The convergence finding is rock-solid.
- **Credential stack depth jump (T11 5.4×):** Verified at 5.59× (2.85% → 15.97%). Same conclusion.
- **Entry-share rise under combined column, imputed, and YOE proxy:** All three methods independently re-derived (+3.48pp combined, +4.66pp imputed, +6.27pp YOE). The triple agreement on direction is robust.
- **Mid-senior credential vocabulary stripping (T11 / T12):** Confirmed under both T12 uncapped and V1 capped Fightin' Words. `qualifications`, `degree`, `bachelor`, `required`, `requirements` all appear in the 2024-favored top-30 under both specifications.
- **Mentoring growth is real, management growth is boilerplate (T11):** Verified by direct sampling. Strict mentoring detector is ~100% precision; the naive `hire` indicator is ~28% precision in 2024 and ~8-12% in 2026 (worse than T11 reported). The "IC + mentoring, not management" reframing is fully supported.
- **T15 semantic-convergence null:** Holds robustly under text-source-controlled (LLM-only) subset. NOT a text-source confound. The convergence rejection stands.
- **AI mention quadrupling:** 13.7% → 52.0% verified.

### What V1 changed or qualified

1. **T08's "company-capping reverses the entry-share rise" was not reproducible.** Under V1's sensitivity grid (baseline / cap20 / dedup / dedup+cap20 / aggregator-exclude / top-20-removed), the 2026 entry-share rise survives every variant by at least +1.4pp under the combined column and +5pp under the YOE proxy. The reversal only appears in one narrow corner: `arshkon-only × YOE × cap20`, where the sign flips by -0.79pp (well within noise). **The entry rise is not concentration-driven; it is real.** The "validity threat" framing in the body of this memo is withdrawn.

2. **The 2026 entry rise is primarily driven by genuine large-employer new-grad programs.** V1's company drilling found that under the YOE≤2 proxy (n=4,022, less duplicate-contaminated), the top contributors are Google, Walmart, Qualcomm, SpaceX, Amazon, Microsoft, Meta, Cisco, Visa — all with plausible within-company entry shares (20-75%). This is a substantively cleaner story: the entry-share rise isn't a market-wide shift, it's a concentrated phenomenon at a dozen large tech/industrial employers running new-grad programs.

3. **The combined-column entry pool has ~23% duplicate-template contamination.** Affirm posted the same exact description 25 times; Canonical 22×; Epic 14×; Google, SkillStorm, Uber 4-5×. Six companies account for 23% of the 2026 combined-column entry pool through `description_hash` collisions. **This is a preprocessing dedup gap, not a data interpretation problem.** A handoff document for the engineering team is at `docs/preprocessing-dedup-issue.md`. In the meantime, the YOE proxy is the cleaner metric for entry-level analyses (n=4,022 has much lower duplicate leverage than n=390). The 2024 baseline does not have the same duplicate-rate.

4. **C++ and C# were severely undercounted in the shared tech matrix.** The `\bc\+\+\b` regex cannot match (word boundaries don't lie between `+` and end-of-word), and scraped LinkedIn text contains markdown escapes (`C\+\+`) that break naive tokenizers. Verified corrected mention rates: **C++ 15.6% → 19.0% (+21% growth, NOT a decliner)** and **C# 11.1% → 11.2% (flat, not declining)**. **T14's "5 declining technologies" list is wrong** and must be revised. T14's language community structure is missing a systems community anchor (C++ has Jaccard ~0.18 with embedded, linux, java, python, rust). Wave 3 tasks that depend on the tech matrix should treat c_cpp/csharp as suspect until the matrix is regenerated.

5. **T13's "57% length growth is genuine content expansion" needs the text-source caveat.** Under LLM-text-only subset, length growth is **only +26%** (vs +72% mixed). About half of the aggregate length growth reflects 2024 (91% LLM-cleaned) vs 2026 (21% LLM-cleaned) text-source composition shift. **Length growth is still real, just substantially smaller** than T13 headlined. T13's "core sections grew more than boilerplate" finding is also robust under text-source control — the relative composition story holds, only the absolute magnitude shrinks.

6. **The `agent` keyword indicator is contaminated.** ~25-45% of `agents?` matches in 2026 are non-AI (insurance agents, legal disclosures, robotics autonomy, "change agent" HR idiom). T08's `desc_contains_agent` +2180% is directionally correct but the absolute magnitude is inflated. The `agentic` token alone is ~95% AI-precision and is the cleaner signal for Wave 3 to use.

7. **T12's uncapped Fightin' Words top-30 contains volume-driven artifacts.** Under V1's capped Fightin' Words re-run, the AI vocabulary (workflows, observability, llm, rag, agent, ai-powered, cloud-native) survives but several of T12's other top terms (`accommodation`, `usd`, `values`, `americans`, `marrying`, `experimenting`) drop out — they were Amazon-driven boilerplate, not genuine cross-period shifts. The credential-stripping subset of T12 (`qualifications`, `degree`, `bachelor`) is robust under capping. Treat the AI and credential subsets of T12 with confidence; treat the rest with care.

### V1's most consequential new observation (NOT in Wave 2)

**Within-company entry posting is extremely concentrated.** Only 106 of 1,201 scraped companies with ≥5 SWE postings have ANY entry-labeled postings under the combined column. **91% of companies with substantial scraped presence have zero entry rows.** Entry-level posting is a specialized activity that most tech employers in the scraped corpus do not do (or do not post to LinkedIn). This is itself a substantive finding — it reframes "entry-share trend" as "concentrated activity at a small set of new-grad-program employers" rather than as a market-wide pattern. **This finding belongs in T16's company analysis in Wave 3 and has been added to the expanded T06 spec for future re-runs.**

### Updated emerging narrative (after V1)

The body of this memo proposed: "Between 2024 and 2026, the dominant change in SWE postings was technology ecosystem restructuring around AI/ML. Within this restructuring, posting requirements expanded significantly... entry-level postings absorbed AI requirements faster than the overall corpus..."

This narrative survives V1 verification with three refinements:

1. **The entry-share rise is REAL but is concentrated at large new-grad-program employers**, not a broad market shift. About a dozen large tech employers (Google, Walmart, Qualcomm, SpaceX, Amazon, Microsoft, Meta, Cisco) drive most of the rise, while 91% of companies with substantial presence have zero entry posts. This is a more interesting story than "entry share rose": **most employers have stopped posting (or never posted) entry-level SWE jobs at all, and the visible entry-level activity is concentrated at a small set of companies running formal new-grad programs.**

2. **Length growth is real but smaller than headlined.** +26% under text-source control vs +72% mixed. The "2026 entry postings are now longer than 2024 mid-senior postings" finding from T08 needs to be re-stated under text-source control. The "core sections grew more than boilerplate" finding from T13 holds.

3. **C++ is growing, not declining.** T14's tech ecosystem story has a missing systems community. The AI/ML cluster fusion finding is unaffected (it doesn't depend on C++/C#), but the "5 declining technologies" sub-finding is wrong.

### What this means for Wave 3

- **T16 (company strategies)** is now critical. It needs to characterize the 91% concentration finding, the duplicate-template handful of companies, and the within-company entry profile of the new-grad-program employers. V1 did some of this in its drilling — Wave 3 T16 should expand and formalize.
- **T14's tech ecosystem claims need a partial re-run.** The c++/c# regex is documented in the engineer handoff doc and will be fixed in preprocessing. For the current run, T18 (cross-occupation) and T22 (ghost forensics) may want to verify any tech-matrix-dependent claim against direct LIKE queries.
- **The new T28 (domain-stratified scope changes)** can use T09's archetype labels and the YOE proxy to do clean within-domain analyses without depending on the contaminated native labels or the broken tech matrix.
- **The new T29 (LLM-authored description detection)** is now even more interesting given the +26% vs +72% length finding — the text-source composition shift may itself be a downstream effect of the LLM-tooling adoption hypothesis.

The body of this memo remains useful for context but the V1 reconciliation above should override any specific claims that conflict.

---

## What we learned

Wave 2 is the most informative gate of the exploration so far. Three independent methods converged on the same headline finding, two of the original RQ1-RQ4 hypotheses were confirmed in revised form, one was rejected, and a previously unanticipated finding has emerged as a strong candidate for the paper's lead. Several preprocessing bugs were also surfaced — the most consequential of which (broken c++/c# regex in the shared tech matrix) would have invalidated several derivative claims if not caught.

### 1. The dominant change is technology ecosystem restructuring (CONVERGENT across three methods)

- **T09 (BERTopic on 8K balanced sample):** AI/ML archetype grew +10.96pp from 2024 to each 2026 period — the only large grower. It absorbs share from .NET (-3.5), frontend/web (-2.3), Spring/Java (-1.8), data engineering (-1.1), python/django (-0.9).
- **T12 (BERTopic on different sample, with Fightin' Words):** AI/ML topic grew +11.95pp — essentially the same number from a different sample with different sampling strategy.
- **T14 (technology co-occurrence networks):** AI cluster fusion. In 2024, modern-AI tools (Copilot, Claude/OpenAI APIs, Hugging Face) formed isolated 2-4 node communities. In 2026 they fuse with traditional ML (PyTorch, TensorFlow, pandas, NLP) into a 29-member ML+GenAI mega-community. AI-mentioning posting share grew 2.8% → 26.2% (9.4×).
- **T08 (univariate distribution profiling):** `desc_contains_ai` 8.7%→40.9%, `desc_contains_agent` +2180%, `desc_contains_rag` +5600%, `desc_contains_llm` 1.0%→10.7%, `desc_contains_copilot` 0.08%→4.2%.

Three independent methods (topic model on sample A, topic model on sample B, co-occurrence graph) plus a univariate keyword scan all agree on the same direction and roughly the same magnitude. This is the strongest empirical signal in the exploration so far.

### 2. The market organizes by tech domain, NOT seniority

T09's NMI scores were striking: archetype × primary_language = 0.1119, archetype × period = ~0.04, archetype × combined_seniority = 0.0155, archetype × yoe_junior = 0.0078. **Primary language dominates the embedding 5× over period and 7× over seniority.** The natural latent structure of the SWE posting market is technology/domain, not career level. The RQ1 framing of "junior vs senior restructuring" was looking at the wrong axis.

T15 corroborates this from a different direction: the dominant variation in semantic space is *period*, not seniority. Within-period centroid cosine is 0.99+; between-period is 0.88-0.97; junior vs senior within a period is an order of magnitude smaller.

### 3. Junior scope inflation IS confirmed — but in a specific form, and partially mediated by domain

- **T11 credential stack depth 7 (all 7 categories present): 3.8% → 20.5% (5.4× jump).** Robust to all sensitivities. Largest single-metric structural change in the dataset.
- requirement_breadth +37%, tech_count +34%, scope_count +97%, ai_mention +296%
- **Entry-level scope inflation is robust across all 4 seniority operationalizations × 2 aggregator variants** (+3-4 breadth points per entry posting in every cell). Notably, the YOE≤2/arshkon-only configuration shows the *largest* inflation (+3.94).
- **Entry-level AI mention 15.8% → 50.1%** under the YOE-based entry definition. Entry-level is absorbing AI requirements faster than the overall corpus.
- **T08:** 2026 entry postings (5,048 chars) are now longer than 2024 mid-senior postings (3,352 chars). This is the descriptive footprint.
- **T13:** Length growth is genuine content expansion. Core sections (summary + responsibilities + requirements + preferred) grew +88.9%, accounting for 85.7% of gross growth. Boilerplate grew 976% relatively but only 41% of gross delta. Within-2024 calibration confirms core ratio 2.5-4.7× (above noise).
- **Surprise within T13:** Responsibilities (+85%) grew MORE than Requirements (+60%). The expansion is in scope-of-role text faster than skill-list text.

But T11 also flagged one important domain-mediated nuance from T09: **AI/ML is the only large-growing domain AND it is structurally less junior-heavy** (entry share 6.0% vs sample-wide 7-8%; YOE-junior 17.2% vs 19.3%). So part of the aggregate "scope inflation" picture is a domain composition effect — the market moved toward AI/ML, which has higher requirements and lower entry concentration. Wave 3 needs to decompose: how much of within-domain entry-level scope inflation vs how much is between-domain composition shift.

### 4. The senior archetype shift is REFRAMED, not as originally hypothesized

The original RQ1 hypothesis was that senior roles would shift from people-management language toward AI orchestration language. T11's strict-management deep-dive revealed:

- **The naive "hire/recruit" trigger went 8.4% → 34.5%, but ~85% is boilerplate artifact** — recruiter disclaimers, "hiring process" language, "new hire training," HR policy text. Not real management responsibilities.
- A strict detector (mentor engineers/juniors, hire engineers, direct reports, performance reviews) shows only modest 9.6% → 13.5% real growth.
- **The growth is dominated entirely by "mentor engineers/junior" (4.2% → 10.0%).**
- **Strict people-manager terms all DECLINED:** `people manager` 2.35% → 0.73%, `direct report` 0.41% → 0.32%, `performance review` 0.20% → 0.13%, `manage/lead a team of` 1.5% → 0.5%.

The reframed senior archetype shift is **IC → IC + mentoring**, NOT IC → people-manager. Employers are writing more junior postings AND asking mid/senior ICs to mentor juniors more often. People-management framing is actively declining. T12 corroborates: 2026 mid-senior postings *lose* `qualifications, required, degree, bachelor, responsibilities` from their core sections — credential and ladder vocabulary is being actively stripped.

### 5. Entry share is approximately stable (Gate 1 finding strengthens, but with a critical caveat)

T08 independently replicated the Gate 1 deep-dive finding using the combined best-available column, `seniority_imputed`, and the YOE proxy. Three independent measurement methods agree: entry share rises ~4-6pp from 2024 to 2026. Only the contaminated `seniority_native`/`seniority_final` (arshkon-only baseline) shows a decline.

**However, T08 also flagged a critical sensitivity reversal:** when companies are capped at 20 postings, the direction of the entry-share rise REVERSES. A small number of high-volume companies drives the 2026 entry-share increase. This is both a validity threat and potentially a substantive finding about market concentration. **It needs investigation in Wave 3** — see "Direction for Wave 3" below.

### 6. The semantic convergence hypothesis is REJECTED

T15's semantic similarity analysis was designed to test whether junior and senior postings became more similar over time (a possible interpretation of "junior scope inflation"). Result: **null and possibly diverging**.

- Embedding cosine: junior↔senior change = -0.001 (flat)
- TF-IDF cosine: junior↔senior change = -0.022 (slightly diverging)
- Below within-2024 noise floor by 4×
- Nearest-neighbor analysis: 2026 junior postings retrieve 2024 junior neighbors AND 2024 senior neighbors slightly above baseline; the shift is AWAY from 2024 mid. Mild polarization, not inflation.

**Recommendation: do not cite a seniority-convergence finding from T15.** This kills one of the original RQ1 sub-hypotheses.

---

## What surprised us

1. **The dominant axis of the SWE posting market is tech domain, not seniority.** T09's NMI numbers (0.11 for language, 0.02 for seniority) were larger than any of us expected. The whole RQ1 framing was anchored on a seniority axis that the data treats as second-order.

2. **AI/ML is structurally less junior-heavy.** Within the AI/ML archetype, entry share is 6.0% vs sample-wide 7-8%. So as the market moved toward AI/ML, it composed itself away from juniors *via the domain shift*, not via per-domain junior cuts. This changes the story.

3. **Mid-senior postings actively LOST credential vocabulary.** `qualifications, required, degree, bachelor, responsibilities` were all stripped from mid-senior 2026 core sections (T12). T11 confirmed: education level is below the within-2024 noise floor — effectively stable. Counter to the prior that AI displacement of juniors would raise degree requirements.

4. **The "senior archetype shift" is mentoring, not management.** Strict people-manager terms all declined; mentorship terms grew. The dominant narrative was wrong on the direction.

5. **2026 entry postings are now longer than 2024 mid-senior postings** (T08).

6. **Tech density (per 1K chars) FELL 26% even as tech count rose 60%.** Postings are longer faster than they're packing requirements. This is suggestive (see "LLM-authored descriptions hypothesis" below).

7. **The naive "hire" indicator overstated management growth by ~3-4×.** This is exactly the kind of measurement artifact the analytical preamble warns about, and it was caught by T11's own strict-vs-broad sensitivity. The strict version is the right metric; the broad version inflates by a factor of 3-4.

8. **Top 1% by requirement breadth is entirely 2026.** No 2024 posting clears the absolute top-1% bar — the complexity distribution has shifted fully into 2026 territory.

9. **Within the same exact title "software engineer", entry seniority share jumped 1.2% → 15.3%.** Same string, 12× more entry-labeled. This is striking and supports the labeling-explicitness story from Gate 1.

10. **C++ and C# were almost completely missing from the shared tech matrix** due to a regex bug — `\bc\+\+\b` cannot match because `\b` does not lie between `+` and end-of-word. C++ is actually in 19.2% of scraped postings; the matrix reported 0.5%. Caught by T14 and independently by T12 (which initially reported "C++ disappeared from 2026" until cross-checking with direct LIKE queries).

---

## Evidence assessment

| Finding | Strength | Notes |
|---|---|---|
| AI/ML domain restructuring | **Very strong** | 3 independent methods converge; survives every sensitivity |
| Credential stack depth 5.4× jump | **Very strong** | Robust to all sensitivities; large effect |
| Description length growth +57% (core sections) | **Strong** | T13 section anatomy confirms it's content, not boilerplate; calibration ratio >2.5× |
| Entry share approximately stable (combined column + YOE) | **Strong** | 3 measurement methods agree; replicated independently in Wave 2 |
| Senior shift to mentoring (not management) | **Strong** | Strict-detector deep-dive; all explicit people-manager terms declining |
| Junior labeling explicitness +1.7-1.8pp | **Strong** | T10 + Gate 1 deep-dive replicate; 2.5× jump in within-title entry share |
| Mid-senior credential vocabulary stripping | **Strong** | T12 Fightin' Words shows 5+ credential terms in 2026 LOSER list |
| Entry-level scope inflation (combined + YOE) | **Strong** | Robust across all 4 seniority operationalizations × 2 aggregator variants |
| Aggregate entry rise (un-capped) | **Conditional** | Reverses under company-capping. A small number of high-volume companies drive it. Needs investigation. |
| Semantic convergence between seniority levels | **Rejected** | Below noise floor under both representations |
| The 90% within-company entry decline (T06) | **Withdrawn** | Inputs were contaminated; T16 will re-run with combined column + YOE proxy |

---

## Narrative evaluation

### RQ1 (employer-side restructuring): **Reframed and confirmed (in part)**

- **Junior scope inflation:** CONFIRMED in the form "entry-level requirements grew across multiple dimensions (tech count, scope terms, AI mentions, credential stack depth)." Robust across operationalizations. But the form is "longer postings packed with more requirements" rather than "denser postings."
- **Senior archetype shift:** CONFIRMED but REFRAMED. Not "people-management to AI orchestration"; instead "IC to IC+mentoring, with credential vocabulary stripped." People-management framings actively declined.
- **Junior share decline:** REJECTED. Under correct measurement (combined column + YOE proxy), entry share is approximately stable or slightly rising. The native-label-based decline is a measurement artifact.

### RQ2 (task and requirement migration): **Promising direction**

T12's Fightin' Words analysis surfaces specific terms that moved between seniority levels. This is the cleanest empirical handle on RQ2 and should be expanded in Wave 3.

### RQ3 (employer-usage divergence): **Wave 3 task**

Untouched by Wave 2.

### RQ4 (mechanisms): **Several Wave 2 findings inform interview design**

- The mentoring-not-management reframing
- The labeling-explicitness shift
- The AI/ML domain restructuring
- The "longer postings, lower density" pattern (the LLM-authorship hypothesis)

---

## Emerging narrative (most coherent version)

> Between 2024 and 2026, the dominant change in SWE postings was **technology ecosystem restructuring around AI/ML**. The AI/ML domain — which absorbed isolated AI tools (Copilot, Claude, RAG, LangChain, MCP) and traditional ML (PyTorch, TensorFlow) into a single cohesive technology cluster — grew its share of SWE postings by ~11pp at the expense of frontend, .NET, Spring/Java, and data engineering. The natural structure of the SWE posting market is organized by technology domain, not by career level: domain explains 5× more variance in posting content than period and 7× more than seniority.
>
> Within this restructuring, posting requirements expanded significantly: the share of postings that simultaneously demand all 7 categories of requirements (tech, soft skills, scope, education, YOE, management, AI) jumped from 3.8% to 20.5% — a 5.4× increase. Entry-level postings absorbed AI requirements faster than the overall corpus (15.8% → 50.1% AI mention rate under YOE-based entry definition), and 2026 entry postings are now longer than 2024 mid-senior postings.
>
> The senior tier did not shift toward people-management as one might have expected from "AI displaces juniors, seniors take over coordination." Instead, explicit people-management language (`direct report`, `performance review`, `people manager`) actively *declined*, while mentorship language grew (`mentor engineers/junior` 4.2% → 10.0%). Mid-senior postings also stripped credential vocabulary (`qualifications`, `required`, `degree`, `bachelor`). The senior archetype is shifting toward IC-with-mentoring-responsibility, not toward formal management.
>
> Entry-level posting share — the original headline question — was approximately stable. Methods that depend on LinkedIn's platform-provided seniority labels show an apparent decline, but this is attributable to differential native-label quality across data snapshots: 41% of arshkon's `seniority_native = 'entry'` rows have YOE >= 5 years, vs only 9.5% in scraped, indicating systematic 2024 label noise. Three independent measurement methods (combined LLM+rule column, text-only imputed, and label-independent YOE proxy) all agree that entry share is approximately stable or modestly increasing. Employers also became significantly more likely to put explicit seniority signals in posting titles (junior keyword share nearly doubled).

This narrative is sharper, more interesting, and better supported than the original "junior scope inflation + senior management shift + employer-usage divergence" framing.

---

## Open issues (the validity threats the V1 verification needs to address)

### 1. The entry-share rise reverses under company capping (T08)

A small number of high-volume companies drives the 2026 entry-share rise. **Until we know who they are, we cannot tell whether the rise is real (e.g., a few large employers genuinely posting more entry roles) or a scraping/aggregation artifact (a few companies whose posting volume blew up during the scrape window).** This is the most important open question from Wave 2.

**Action:** Add a Wave 3 sub-task (extending T16) or a Gate 2 micro-investigation to identify the 20 companies driving the entry rise, profile their posting characteristics, and determine whether the rise is broad-based or concentration-driven.

### 2. The C++/C# regex bug invalidates several T14 derivative claims

Verified by both T14 and T12: `swe_tech_matrix.parquet` reports c_cpp at 0.5% and csharp at 0.15%, but direct queries show C++ at 19.2% and C# at 11.3%. The bug has two root causes: (a) `\bc\+\+\b` cannot match in regex because `\b` does not match between `+` and end-of-word; (b) scraped LinkedIn text contains markdown-escaped characters (`C\+\+`, `C\#`) that break naive tokenizers.

**Affected:** T14's "5 declining technologies" list (C++/C# may have actually grown), the language community structure (C++ should anchor a systems community), and any T11/T08 metric that uses the tech matrix as input. Re-running with a corrected regex is required before Wave 3.

**Root-cause analysis (for prompt improvements):** The bug exists because Agent Prep wrote regex patterns without validating them against an independent reference. T14 caught it via the structured-vs-extracted validation step (Spearman ρ=0.77 was high but not perfect — the missing C++/C# was visible in the structured baseline). T12 caught it independently when its initial result said "C++ disappeared from 2026," which the agent correctly recognized as implausible and cross-checked.

**Holistic prompt improvement (proposed):** Make regex/pattern validation against an independent reference a *core* preamble principle, not an analytical-preamble principle. Currently the analytical preamble (Wave 2+) requires sampling 50 matches per pattern, but Agent Prep is in Wave 1.5 and only sees the core preamble. The shared preprocessing spec should also explicitly require: "When constructing regex patterns or extractors that downstream agents will depend on, validate them against an independent reference (structured fields, manual sample, or direct LIKE queries) before saving to the shared artifact."

### 3. swe_cleaned_text.parquet has stopwords stripped, breaking phrase analyses

Agent H (T13/T12) discovered that the shared cleaned text artifact has English stopwords removed, which destroys phrase-level analyses (tone markers like "you will", section header detection like "What you'll do," management phrases like "lead a team"). T13 worked around it by reading raw text directly from `unified.parquet`. Documented in `shared/README.md` after the fact.

**Root cause:** Agent Prep was instructed to "remove standard English stopwords" without considering downstream phrase-matching use cases. The transformation is destructive and irreversible from the artifact alone.

**Holistic prompt improvement (proposed):** Add a principle to the shared preprocessing spec: "Lossy text transformations (stopword removal, stemming, lowercasing) destroy information that downstream agents may need. Shared text artifacts should preserve maximum information; downstream-specific transformations should be applied locally by the agents that need them." Or alternatively: "Save multiple text variants if downstream uses diverge — e.g., a `text_clean` column for phrase-preserving analyses and a `text_tfidf` column for bag-of-words analyses."

### 4. Arshkon entry pool is finance/banking-skewed (T12)

T12 found that arshkon's entry-level term comparison is dominated by finance terms (`laundering, vba, macros, econometric, liabilities`). The contamination isn't just YOE noise — it's domain composition skew. This affects any cross-period entry-level term comparison that uses arshkon as the 2024 baseline.

**Mitigation already in place:** The seniority operationalization rule already directs agents to use the YOE proxy as a co-equal validator and to investigate sensitivity disagreements. The finance contamination is a specific instance of this. T11's domain-stratified scope inflation step (deferred, awaiting T09 labels) will partially address it.

**Holistic prompt improvement (proposed):** Strengthen the cross-source comparison principle: "Before drawing conclusions from any cross-source text comparison, profile the industry/sector composition of each source. Differential industry mix can drive false 'temporal' signals." This is partially in T05 already but should be a general principle.

### 5. Open-ended text comparisons are dominated by Amazon-style company concentration (T12)

Several top scraped bigrams in T12 are Amazon-specific boilerplate (`destination stage`, `amazon.jobs how-we-hire`, `empowers amazonians`). The company-capping sensitivity (b) catches this, but Agent H ran the primary analysis uncapped.

**Holistic prompt improvement (proposed):** Promote company capping from "essential sensitivity" to **primary specification** for any open-ended corpus comparison (Fightin' Words, BERTopic on full corpus, term frequency rankings). The current framework treats it as a sensitivity check; for these task types, it should be the default.

### 6. Several "data quality" anomalies are NOT worth fixing (per user steer)

The user noted that we should focus on what our research actually needs and not fix unnecessary labels. Applied:

- `is_remote_inferred` is broken (0/63K). **Verdict: do not fix.** We don't need it for the research; the source `is_remote` flag is sufficient for 2026 (only period where it's populated), and remote-status analysis was never a core RQ.
- Volunteer job contamination in asaniczka (T15). **Verdict: accept as classifier imperfection.** Low prevalence; doesn't affect aggregate findings.
- Aggregator "decline" 24.6%→14.3% is mostly an asaniczka composition artifact. **Verdict: noted as a sensitivity, not a fix needed.** Aggregator-exclusion sensitivity already in the framework.
- NYC scraper coverage 28x growth (66→1,862). **Verdict: noted as a scraper coverage difference.** Affects geography stratification but is the actual 2026 frame, not a bug.
- Industry label format harmonization (compound vs single labels). **Verdict: needed for industry-stratified analysis.** Should be addressed in T18 or as a Wave 3 sub-task — industry is genuinely useful for cross-occupation analysis (T18).
- BERTopic-surfaced Dice/defense-contractor templates not stripped by Stage 3. **Verdict: minor; does not affect headline findings.** Could be addressed in a future preprocessing pass.

### 7. The hire-indicator boilerplate question (user asked)

The user asked where the "hire" boilerplate came from given that boilerplate removal is supposed to handle this. **Answer:** T11 ran on `description_cleaned` (LLM-cleaned where available, falls back to rule-based). The boilerplate stripping in Stage 3 (rule-based) and Stage 9 (LLM) targets specific section types: benefits, about-company, legal/EEO. They do NOT target HR-process language like "our hiring process," "new hire training," recruiter signature blocks, because that language can appear inside legitimate sections (e.g., a requirements section saying "experience hiring engineers" or a responsibilities section saying "help shape our hiring process"). The boilerplate stripping is doing its job; the keyword indicator was just naive.

This is actually a useful diagnostic: it tells us that 2026 postings have substantially more HR-process language *inside the substantive sections*, not in benefits boilerplate. That's a real change in posting content (recruiters writing more about themselves, the hiring process being described in more detail) — it's just not "management language."

**Holistic prompt improvement (proposed):** When constructing keyword indicators, the analytical preamble should require not just sampling 50 matches but also reporting WHICH SECTIONS the matches come from (using T13's section classifier). A pattern that fires mostly in benefits sections is structurally different from one that fires in requirements sections.

---

## The LLM-authored descriptions hypothesis (proposed new task)

The user raised an excellent hypothesis: **the description length growth, the tech-density decrease, and the verbose/uniform tone may all be downstream effects of recruiters using LLMs to draft job descriptions.**

This is testable. Signals that would support it:
- **LLM signature vocabulary:** classic LLM tells like `delve`, `tapestry`, `robust`, `leverage`, `unleash`, `elevate`, `embark on`, `navigate the`, `cutting-edge`, `in the realm of`, `it's important to note`, `furthermore`, `moreover`, `comprehensive`, `seamless`
- **Em-dash frequency:** LLMs use em-dashes much more than humans
- **Sentence length distribution:** LLMs produce longer, more uniform sentences
- **Vocabulary diversity (Type-Token Ratio):** if LLMs are writing them, descriptions may become MORE uniform across postings (lower cross-posting vocabulary diversity)
- **Bullet point structure:** LLMs love structured bulleted lists
- **Boilerplate-like phrases inside requirements:** "Strong communication skills", "passion for [X]", "fast-paced environment", "self-starter mentality" — these are LLM-generation hallmarks

If true, this hypothesis would unify several Wave 2 findings into one mechanism:
- The 57% length growth (LLMs are verbose, no typing cost)
- The 26% tech-density decrease (more padding per technology)
- The +296% AI mention growth (recruiters who use AI tools also write about them)
- The compressed entry-vs-mid-senior readability gap (LLM-generated text has uniform readability)
- The "hire" boilerplate inside substantive sections (LLMs include recruiter-like framing naturally)
- The credential vocabulary stripping in mid-senior (LLMs prefer more flowing prose to bullet-point credentials)
- The AI vocabulary explosion (literal LLM-self-reference would mention AI tools)

It would also have a substantive implication: **part of what we're measuring as "employer requirements changing" may actually be "recruiters' tools changing."** This is methodologically important and potentially a paper finding in its own right ("the AI-coding-tool adoption window is also the AI-job-description-writing-tool adoption window, and longitudinal posting analyses must account for this").

**Action:** Add a Wave 3 task to measure LLM-authorship signals across the corpus. See "Direction for Wave 3" below.

---

## Direction for Wave 3

Wave 3 was originally scoped as T16-T23 (company strategies, geography, cross-occupation, temporal patterns, seniority boundaries, senior evolution, ghost forensics, employer-usage divergence). Given Wave 2 findings, the priorities shift:

### Modifications to existing Wave 3 tasks

- **T16 (Company hiring strategy typology):** Critical priority. Must investigate the company-capping reversal of the entry-share rise. Identify the 20 companies driving the 2026 entry rise; profile their posting characteristics; determine whether the rise is broad-based or concentration-driven. Re-run the within-company decomposition under the combined seniority column AND the YOE proxy (replacing T06's contaminated-input version).
- **T18 (Cross-occupation boundary):** High priority. The AI/ML restructuring narrative depends on whether the same pattern appears in control occupations. If control also shows AI-mention growth, the SWE-specific framing weakens.
- **T19 (Temporal patterns):** Medium priority. Check whether the company-capping reversal also reverses under within-March and within-2026-04 stability checks — if a few companies posted heavily on specific scrape days, that's a different artifact.
- **T20 (Seniority boundary clarity):** Reframed. T15 already showed that semantic seniority convergence is null; T20 should focus on the *feature-based* boundary (YOE distribution overlap, requirement breadth overlap) rather than the semantic boundary, and stratify by domain archetype.
- **T21 (Senior role evolution):** High priority. The mentoring-not-management reframing from T11 is one of the strongest Wave 2 findings. T21 should validate this with a strict-detector approach from the start.
- **T22 (Ghost forensics):** Reframed. The AI ghostiness question is more interesting now that we know AI mentions exploded. T22 should also check whether AI requirements are concentrated in companies with LLM-authored-looking descriptions (links to the LLM-authorship hypothesis).
- **T23 (Employer-usage divergence):** Standard.

### Proposed new Wave 3 tasks

**T-NEW1: Company concentration drilling for the entry-share rise.** Could be folded into T16 or run as a standalone mini-task before Wave 3 dispatch. Identifies the specific companies driving the unstable entry-share rise; profiles their post characteristics (titles, industries, descriptions, posting volume timing); determines whether the rise reflects real labor market change or scraper/aggregator concentration.

**T-NEW2: LLM-authored description detection.** Detect signals of LLM authorship in posting text (signature vocabulary, em-dash density, sentence length uniformity, vocabulary diversity, bullet structure). Compute the prevalence of likely-LLM-authored descriptions by period and by company. Test whether the description length growth, the tech-density decrease, and the AI mention explosion correlate with LLM-authorship signals at the posting level. This could become the paper's most novel methodological contribution.

**T-NEW3: Domain-stratified scope changes (re-run T08 step 7 + T11 step 7).** With T09 archetype labels now available, re-run the deferred decomposition steps. Specifically: how much of aggregate "scope inflation" is within-domain vs between-domain composition shift? How do junior vs senior requirements compare *within* each domain archetype?

### Priorities for V1 verification (Gate 2)

The V1 agent should adversarially re-derive and validate:

1. **The AI/ML archetype growth** (+10.96pp from T09, +11.95pp from T12). Independent re-derivation; should match within 5%.
2. **The credential stack depth jump** (3.8% → 20.5%). Recompute from scratch; check the extractor's correctness.
3. **The combined-column entry-share rise** (under at least 3 measurement methods).
4. **The C++/C# regex bug fix.** Recompute the tech matrix with corrected regex; confirm c_cpp is now ~19% and csharp ~11%; identify which T14 findings change.
5. **The mentoring vs management strict-detector finding** (T11 §5). Re-validate by sampling 50 matches per term in the strict set.
6. **The text_source confound flag from T15.** Quantify how much of T15's "no convergence" finding could be driven by 2024 sample being 94% LLM-cleaned vs 2026 being 79% rule-cleaned.
7. **Open-ended text comparison with company capping as primary** (re-run T12's Fightin' Words with capped corpus, see if Amazon-specific bigrams disappear and core findings persist).
8. **Section classification consistency** between scraped-markdown and Kaggle-flat-text formats (T13's section classifier handles both, but verify it's not systematically miscounting one format).

V1 should NOT:
- Re-run the full Wave 2 task suite — it's a verification, not a re-do
- Replace any agent's analysis — its job is adversarial QA on the headline findings

---

## Holistic prompt and task improvements (for the next exploration re-run)

The user's framing principle is important: don't bolt on instance-specific instructions; make holistic changes that nudge all agents toward higher-quality work in any future run. Based on Wave 2 findings, the improvements should target principles rather than specific patterns. Six proposed changes, all of which we should apply to `docs/preprocessing-schema.md`, `docs/task-reference-exploration.md`, and `docs/prompt-exploration-orchestrator.md` after the user reviews:

### Improvement 1: Pattern/extractor self-validation (CORE preamble, all waves)

Currently the analytical preamble (Wave 2+) item #6 requires sampling 50 matches per keyword pattern. This should be elevated to the **core** preamble so it applies to Wave 1 and Wave 1.5 agents too, and extended:

> **Pattern validation.** Any time you construct a regex, keyword indicator, or extractor — especially one that downstream agents will depend on — validate it against an independent reference before publishing. Independent references include: (a) structured fields in the data (e.g., `skills_raw` in asaniczka, native columns), (b) direct SQL LIKE/REGEXP queries on the raw text, (c) manual sample of 50 matches with precision assessment, (d) cross-checking against another extraction method. If the pattern's count diverges materially from an independent reference, the pattern is broken — fix it before saving the result. Patterns that look syntactically correct but are silently broken (e.g., `\b` near non-word characters) are a known failure mode; never trust a regex without empirical validation.

This would catch the c_cpp/csharp bug at the source.

### Improvement 2: Shared artifact production principles (core preamble + shared preprocessing spec)

Shared artifacts that other agents will load and trust deserve extra care. The shared preprocessing spec should add:

> **Shared artifact discipline.** When you produce an artifact other agents will load:
> 1. **Validate every transformation against an independent reference** (per the pattern validation principle above).
> 2. **Preserve maximum information.** Lossy transformations like stopword removal, stemming, lowercasing destroy information that downstream agents may need but cannot recover from the artifact. If different downstream uses need different transformations, save multiple variants OR save the least-transformed version and let downstream agents apply their own destructive cleanups locally.
> 3. **Document supported and unsupported uses.** The artifact's README must state explicitly which downstream analyses the artifact supports (e.g., "TF-IDF and topic modeling") and which it does NOT support (e.g., "phrase matching, tone marker detection, section header detection — for these, read raw text directly from unified.parquet").
> 4. **Document known limitations and bugs at the time of publication.** If a regex was validated against ground truth and found to have <90% recall on some category, document it; downstream agents need to know.

This would have caught the stopword-stripping issue and forced clearer documentation.

### Improvement 3: Section-aware keyword interpretation (analytical preamble, refines #6)

The hire-indicator boilerplate finding shows that *where* a keyword fires matters. Strengthen item #6:

> **Keyword indicator validation.** Sample 50 matches per pattern and assess precision. **Additionally, when a section classifier is available (T13), report the section distribution of matches.** A pattern that fires mostly in benefits/legal/about-company sections is structurally different from one that fires in requirements/responsibilities. Report results both for the full text and for the requirements/responsibilities subset.

### Improvement 4: Open-ended corpus comparisons need company capping as primary (sensitivity framework, dimension b)

Currently dimension (b) "company capping" is listed as a sensitivity. For task types where uncapped corpus comparisons are systematically dominated by the largest few employers (Fightin' Words, BERTopic on full text, term frequency rankings), capping should be the **primary specification** with uncapped as the alternative. Update the sensitivity framework dimension (b):

> (b) **Company capping.** Primary depends on task type. For per-row comparisons (entry share, AI prevalence, length), uncapped is primary and capping is the alternative. For corpus-level comparisons (Fightin' Words, BERTopic on full corpus, term frequency rankings, embedding centroids), **capping at 20-50 postings per `company_name_canonical` is the PRIMARY specification** because uncapped results are systematically dominated by the largest few employers. Uncapped is the alternative for these tasks.

### Improvement 5: Cross-source industry/sector profiling before text comparisons (analytical preamble, new principle)

The arshkon finance contamination is a specific instance of a general issue. Add to the analytical preamble:

> **Cross-source composition profiling.** Before drawing conclusions from any cross-source text comparison (term frequency, topic distribution, distinguishing terms), profile the industry/sector/company composition of each source. Differential composition can produce false "temporal" signals. If composition differs materially, either restrict to the comparable subset or report the composition difference alongside the text difference and discuss whether the text difference holds within matched composition cells.

### Improvement 6: Sensitivity disagreement is a finding to investigate, not just report (sensitivity framework)

The current framework says "When materially sensitive, INVESTIGATE WHY." This is good, but Wave 2 produced a case (the entry-share company-capping reversal) where the sensitivity is the most important finding in the result, and the agent reported it but didn't drill in. Strengthen:

> **Materiality and investigation.** A finding is materially sensitive to a dimension if the alternative specification changes the main effect size by >30% or flips the direction. **When a finding IS materially sensitive, the sensitivity is itself a finding that requires deeper investigation, not just a flag in the report.** At minimum, the agent must: (a) identify the specific rows/cells/companies/terms that drive the difference between specifications, (b) characterize what they have in common, (c) state the most likely mechanism, (d) recommend a follow-up analysis that would resolve the question. A bare "this finding is materially sensitive to dimension X" without drilling in is insufficient.

---

## What to do before Wave 3 dispatch

1. **Apply the holistic prompt improvements (1-6 above) to the schema, task reference, and orchestrator prompt.** These are general principles that improve the next exploration re-run. Wait for user approval.
2. **Dispatch V1 verification** to adversarially re-derive the top Wave 2 findings, fix the c++/c# regex bug, and investigate the company-capping reversal.
3. **After V1 returns:** Update INDEX.md with verified findings, decide which (if any) Wave 2 results need to be withdrawn or re-stated.
4. **Then dispatch Wave 3** with the modifications and new tasks proposed above.

---

## Current paper positioning

The paper positioning has solidified significantly. The strongest framing now has three substantive contributions plus a methodological one:

**Lead empirical finding:** The technology ecosystem of SWE postings restructured around AI/ML between 2024 and 2026. AI/ML grew +11pp at the expense of frontend, .NET, Spring/Java, and data engineering. The natural latent structure of the SWE posting market is organized by tech domain, not career level (validated by NMI: domain explains 5× more variance than period and 7× more than seniority).

**Second empirical finding:** Entry-level requirements expanded substantially. The share of postings demanding all 7 categories of requirements grew 5.4× (3.8% → 20.5%). 2026 entry postings are now longer than 2024 mid-senior postings. Entry-level AI requirements grew from 15.8% to 50.1%.

**Third empirical finding:** The senior tier shifted toward IC-with-mentoring rather than toward people-management. Explicit people-management language declined; mentorship language grew. Mid-senior postings stripped credential vocabulary.

**Methodological contribution:** Cross-temporal comparisons using platform seniority labels (LinkedIn `seniority_native`) are unreliable when label quality differs across data snapshots. We present a measurement framework (combined LLM+rule routing column, validated by label-independent YOE proxy) and document how an apparent 8pp entry-share decline becomes approximately stable under correct measurement. We also document that employers' explicit signaling of seniority increased substantially (junior keyword share nearly doubled) — a separate finding about how the labor market data ecosystem evolved alongside the labor market itself.

If the LLM-authored-descriptions hypothesis (proposed Wave 3 task) is confirmed, a fifth contribution may emerge: longitudinal posting analyses must account for the parallel adoption of LLM-based JD-writing tools, which can inflate apparent content changes. This would be a methodological warning of broad applicability.

If we stopped here, the paper would already have a coherent narrative and four credible findings. Wave 3 will primarily strengthen existing findings (T16 company drilling, T18 cross-occupation validation, T22 ghost analysis on AI requirements) and add the LLM-authorship analysis as a potential new contribution.
