# Data and Methods — Slide-Ready Outline

*Each subsection follows: **Question** → **Considered** → **Chose** → **Why**. Numbers and citations inline so individual blocks copy cleanly into slides.*

---

## Data

### 1.1 Sources

**Question.** Where did the postings come from?

**Three LinkedIn-only sources, all publicly available:**

- **Kaggle [arshkon](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)** — 2024-04-05 → 2024-04-20; 4,691 SWE rows; periodic scrape with structured-skills join.
- **Kaggle [asaniczka](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024)** — 2024-01-12 → 2024-01-17; 18,129 SWE rows; one-week scrape via third-party API.
- **Our scrape** — daily 2026-03-20 → 2026-04-18; 45,317 SWE rows; 26 US metros; uses [`jobspy`](https://github.com/cullenwatson/JobSpy) against LinkedIn public job-search pages. No authentication, no personal data beyond public posting content. Daily cron on EC2; outputs to private S3.

**Why these three:**

- LinkedIn dominates US tech-employer recruiting.
- All three publicly available — paywalled datasets exclude replication.
- Two independent 2024 sources let us run a within-2024 placebo.
- Project commitment: open data in, open data out.

**Limitations.** Kaggle authors don't publish full crawl methodology — cited directly. 22-month gap to 2026 — no overlap period.

**Open queries used in our scrape:** documented verbatim in appendix.

### 1.2 Cross-source comparability

**Question.** Are the sources comparable enough to pool?

**Within-2024 placebo (arshkon vs asaniczka):**

- SWE share, title, seniority, YOE distributions — comparable.
- Geo coverage broadly aligned (broad-metro both).
- Description length comparable.
- *ML Engineer cross-source asymmetry: 78% SWE-share arshkon vs 59% asaniczka — stratify, do not pool, for ML-specific claims.*

**2024 vs 2026:**

- 2024 baseline restricted to 26-metro subset matching scrape coverage.
- Control occupations stable (counterfactual check).
- Per-day / per-week posting volume aligned (corrected for market conditions).
- Tech-word density on stable terms (Python, Java, Linux, Git, SQL, Docker, K8s, JS, TS, AWS, REST) — 0.0% decay across all eleven anchors.

**Calibration.** Within-2024 SNR — AI-strict 32.9, scope-terms 42.8 (clean signals); entry-share ≈ 1 (near noise floor; reported with that floor explicit).

### 1.3 Schema unification

**Question.** How do we get one canonical row across three different scrape formats?

- **Considered:** lowest-common-denominator (drop fields any source lacks); per-source fork (run analysis three times).
- **Chose:** 39-column canonical schema; cast every source's columns to it; `source` column preserves provenance.
- **Why:** lowest-common-denominator throws away signal; per-source fork makes cross-period comparison impossible.

**Source-asymmetric fields (named):** `company_industry`, `company_size` 0% on asaniczka; `skills_raw` 0% on scraped; `date_posted` 1% on scraped (use `scrape_date`); `is_remote` 0% on 2024.

### 1.4 Aggregator handling

**Question.** How to handle postings authored by job-board / staffing-firm intermediaries?

- **Considered:** drop entirely; treat as direct employers; flag and stratify.
- **Chose:** `is_aggregator` flag + stratified reporting + `company_name_effective` extractor for end employers where recoverable.
- **Why:** Dropping loses 14% of corpus, concentrated in secondary metros (Charlotte 43% aggregator-mediated). Pooling biases firm-level inference.

**Known limit.** End-employer extractor recovers canonical big-tech end-clients in <1% of aggregator SWE postings. Naming is misleading; flagged for fix.

### 1.5 Dedup

**Question.** How to remove duplicate postings (multi-location, daily re-scrapes, cross-source overlaps)?

- **Considered:** `job_id` exact match only; title-text fuzzy; multi-stage waterfall.
- **Chose:** waterfall — `job_id` → exact-opening hash → fuzzy `token_set_ratio ≥ 85` → multi-location collapse.
- **Why:** Each stage catches a different failure mode; per-stage drop rates are auditable.

**Drop rate.** 1,952,715 → 1,452,875 (~26%). Indeed alone absorbs 46.7% of dedup loss (multi-location collapse + daily re-scrape overlap).

### 1.6 Classification (rule-based + LLM)

**Question.** How to assign SWE / Control occupation, seniority, YOE per posting?

- **Considered:** pure regex / dictionary (`stage5_classification.py` reached 1,709 lines, 37 named regex patterns, before deprecation); pure LLM; rule + LLM-fallback (chosen); embedding similarity (cross-check, not primary).
- **Chose:** rule-first per task; LLM where rule abstains or is known to fail; both stored, LLM canonical.
- **Why rule first:** cheap, deterministic, audit trail. LLM is calibrated against rule, not vice versa.

**Why LLM canonical — three failure modes the rule cannot resolve:**

- *Boilerplate is not regex-shaped.* EEO / "About us" / benefits / recruiter CTAs are firm-specific and front-loaded; length-truncation damages signal. **5,980 postings (23.9% of raw AI-mentioning) lose the AI mention after Stage-9 boilerplate stripping** — those mentions lived only in non-core text.
- *SWE classification needs description, not title.* Rule ↔ LLM agreement 80.6%; disagreements concentrate in systems / test / reliability / product / electrical / FAE titles. Closing the gap by regex would require a per-firm per-year title taxonomy — itself a labelling task.
- *YOE multi-mention path resolution is structurally beyond regex.* Rule rejects company-history / salary / age traps competently. Cannot resolve OR-joined ladders ("15 yrs Bachelors / 10 Masters / 8 PhD" → 8) or AND-joined required-bullet lists (11 bullets, 2–8 yrs each → binding floor 8).

**Two-stage LLM design:**

- **Stage 9** strips boilerplate sentence-by-sentence → `description_core_llm` (mean 40.4% word reduction; raw 590 → cleaned 328; p10/p50/p90 = 8% / 42% / 69%).
- **Stage 10** classifies / extracts on cleaned text — easier input for the same model.

**Validation:**

- *Internal:* rule ↔ LLM SWE agreement 80.6%; seniority κ = 0.81 (arshkon, scraped), 0.51 (asaniczka, mid-senior-dominated).
- *Owed before submission:*
  - Per-class human-vs-LLM table (precision / recall / F1 / κ) on stratified ≥300-instance set, ≥2 annotators ([Calderon et al. ACL 2025](https://aclanthology.org/2025.acl-long.782/) alt-test).
  - Subgroup accuracy of that table (company-size, metro tier, posting length, seniority cell) ([Boyeau et al. EMNLP 2024](https://aclanthology.org/2024.emnlp-main.536.pdf)).
  - Refusal / parse-failure rate by class and source.
  - Prompt-paraphrase ablation ([ProSA, Hu et al. 2024](https://aclanthology.org/2024.findings-emnlp.108/)).
  - Second-model rerun on 5–10k subsample.

### 1.7 Analysis frame — the 110k balanced core

**Question.** Which subset carries the headline analyses?

- **Considered:** all available rows (1.45M); random 100k sample; stratified balanced sticky core.
- **Chose:** stratified sticky core, 110k. LinkedIn × English × ≥15 words × assigned `analysis_group ∈ {SWE, Control}`, stratified over `source × analysis_group × date_bin`.
- **Why 110k:** T07 power analysis says it suffices for every headline at junior (YOE ≤ 2) and senior (YOE ≥ 5) cuts. Beyond that, SWE-junior on 2024 data is exhausted; additional rows would be predominantly Control, diluting SWE comparisons.

**Coverage.** Kaggle ~100% in-frame; our scrape 56% in-frame by design (44% held back so 2026 doesn't swamp 2024).

**Why "sticky".** Same row IDs across analyses (deterministic stratified sample seeded once); ensures every paper-level statistic is cross-comparable.

**Frame composition (after refactor):** ~80% SWE / ~20% Control. *No SWE-adjacent tier — see §2.1.*

### 1.8 Known gaps & biases

- **22-month observation gap.** Single biggest temporal confound. Mitigated by within-period 2024 placebo.
- **Platform drift.** LinkedIn UI and `seniority_native` schema changed 2024 → 2026; some shifts may be platform relabeling.
- **Platform selection.** LinkedIn-only primary; Indeed sensitivity-only.
- **Survivorship bias.** Postings that stayed up long enough to scrape may differ from short-lived.
- **Geographic.** 26 metros only in 2026; non-target states are zero by design.
- **Seniority signal.** LLM abstains on 34–53% of in-frame rows; arshkon `entry` median YOE 4 (not clean); asaniczka has no `entry` value.
- **Aggregator share by source.** asaniczka 16.2% / arshkon 9.2% / scraped 17.0%.
- **ML Engineer asymmetry.** 78% SWE-share arshkon vs 59% asaniczka — stratify, do not pool.
- **Description-length floor.** <15-word postings excluded; ghost / inflated postings flagged via `ghost_assessment_llm`.

---

## Measurement / Methodology

### 2.1 SWE vs Control definition (refactored to two-way)

**Question.** How do we draw the SWE / Control line?

- **Considered:** three-way SWE / SWE-adjacent / Control; binary SWE / Control.
- **Chose:** binary SWE / Control. SWE-adjacent (data eng, ML eng, security, technical architects) folded into SWE on description-level evidence.
- **Why:** the SWE-vs-adjacent line was a description-judgment call that didn't correspond to a stable employer construct. Reviewers asked what the adjacency added. We collapsed it; the operationally meaningful boundary is software-or-not.

**SWE.** Postings whose primary function is software engineering: software engineer / developer, SWE, full-stack, front-end, back-end, language-suffixed roles (Python engineer), data engineering, ML engineering, platform / SRE / security engineering when description-confirmed. Caught by `SWE_INCLUDE` regex + `swe_classification_llm = 'SWE'` + embedding similarity ≥ 0.85.

**Control.** Non-software occupations posted by the same kinds of employers, used as counterfactual for whether AI-vocabulary diffusion is software-specific: civil and mechanical engineers, nurses, accountants, marketers, sales. Caught by `CONTROL_PATTERN`. Distinguishes AI-as-job-content from AI-as-recruiting-boilerplate.

### 2.2 Modelling — role taxonomy + skill content + discovery (the hybrid framework)

**Question.** How do we measure year-over-year change in *what postings demand*?

**Considered:**

- *Pure topic modelling* (BERTopic, NMF) on full corpus. Tested in S27_v2: 95 BERTopic topics, 36% noise, ARI 0.64. Useful for discovery; *not* for "Full Stack 2024 vs Full Stack 2026" — cluster boundaries shift across separate fits, c-TF-IDF top-words are dominated by vocabulary turnover (86% novel terms in 2026), and cross-period mapping by name is brittle.
- *ESCO sentence-level skill extraction* (Chen, Sun & Yuan 2026). Methodologically the gold standard but 150k–300k LLM calls — too heavy for an AIES timeline.
- *LaborSpace-style task-graph approach* ([Feng, Wachs, Daniotti, Neffke 2023, arXiv:2311.06310](https://arxiv.org/pdf/2311.06310)). Builds a software-task taxonomy from Stack Overflow via bipartite stochastic block model. Conceptually relevant; their corpus is a different substrate (Stack Overflow), so the taxonomy doesn't transfer. We borrow the *idea* — fixed taxonomy + discovery layer — without the specific construction.
- *Hybrid: fixed role taxonomy + lightweight skill tagging + BERTopic for discovery (chosen).*

**Chose.** Three-layer hybrid:

- **Layer 1 — Role-family taxonomy (the comparison axis).** ~12 parent role families defined a priori before looking at the diffs:
  - Frontend / Full-Stack Web
  - Backend / API / Distributed Services
  - Mobile (iOS / Android)
  - Embedded / Systems / Firmware
  - Data Engineer / Data Platform
  - ML Engineer (traditional, pre-LLM)
  - AI / LLM Engineer (RAG, agents, prompt engineering — the new emergent family)
  - DevOps / SRE / Platform
  - Security Engineer
  - QA / Test
  - Forward-Deployed / Solutions / Customer Engineer
  - Legacy stack (.NET / Java EE / mainframe — folded together because the question is "is this family shrinking?")
  - One LLM call per posting against fixed prompt; same prompt 2024 and 2026.
- **Layer 2 — Skill-tag list (the content axis).** ~50 specific skills pre-registered before classification: LLM API integration, RAG / vector search, prompt engineering, agent orchestration, model evaluation, code review, system design at scale, mentorship, agile / scrum, incident response, data-pipeline authoring, Kubernetes, frontend framework, mobile native, cloud infrastructure, legacy stack markers, etc. One LLM call per posting tags the binary skill matrix.
- **Layer 3 — BERTopic for discovery.** Fit on pooled 2024 + 2026 corpus on `description_core_llm`. Used to (a) validate the role-family taxonomy, (b) detect topics that don't fit any parent family — emergent role candidates, (c) discover within-family sub-archetypes.

**Why hybrid:**

- Fixed taxonomy makes "Full Stack 2024 vs Full Stack 2026" a meaningful comparison; pure clustering does not.
- 50-skill tagging is interpretable, replicable, and ~$50–150 in LLM cost on the 110k frame; ESCO sentence-level extraction is overkill at AIES rigor.
- BERTopic stays in the pipeline as discovery, not inference.

**Embeddings (for BERTopic discovery layer).** OpenAI `text-embedding-3-large` at `dimensions=1024` on `description_core_llm`. *Considered:* `all-MiniLM-L6-v2` (256 ctx, 22M params — used in S27_v2 pilot but truncates long JDs); open-weight upgrades (BGE-large, Nomic 8K-context); Voyage and Gemini APIs. Chose OpenAI for already-configured infrastructure; the dominant quality jump is 256 → 8K context, which any modern embedder solves.

### 2.3 Within-firm panel — the identification anchor

**Question.** How do we separate compositional change (different firms posting) from mutational change (same firm rewriting)?

**Considered:** aggregate cross-period only; within-firm panel; within-firm-within-title matched-pair panel.

**Chose.** All headline claims reported at three levels:

- **Aggregate.** Full balanced 110k frame, 2024 vs 2026. Compositional + mutational mixed.
- **Within-firm panel.** 503 firms with ≥3 SWE postings in both eras inside the balanced core (18,787 postings). Within- vs between-firm components reported separately.
- **Within-firm-within-title matched panel.** Same firm + same `title_normalized` posted in both eras. Smaller (~5–15k pairs) but cleanest available identification — the [Atalay et al. 2020](https://www.aeaweb.org/articles?id=10.1257%2Fapp.20190070) within-title task-change template.

**Why three levels.** A finding that holds at all three is robust to composition, firm-mix, and title-rebrand confounds. A finding that holds only at aggregate is a composition artefact and we say so.

### 2.4 Robustness

Every headline is re-run under:

- Aggregator inclusion / exclusion.
- Cap / uncap postings per company (cap = 30 default).
- Within-firm panel vs full analysis frame.
- Alt-classifier sensitivity (rule-only, LLM-only, intersection, disagreement subset).
- Alternative seniority definition: direct (`seniority_final`) vs YOE-based (J3 = ≤2 / S4 = ≥5) vs intersection.
- Alternative YOE bands (J ≤1 / 2 / 3; S ≥5 / 6 / 7).
- Within-period 2024 placebo (arshkon ↔ asaniczka) — headline effects ≈ 0 expected.
- Bootstrap 95% CIs on every headline rate.
- Effect-size reporting alongside significance.

### 2.5 Vocabulary lists — the regex layer

**Question.** What counts as an "AI mention" or a "scope marker" or a "process-scaffolding marker"?

**AI-rate regex — two tiers, both calibrated against within-2024 SNR:**

- *Broad* — includes ML, deep learning, Copilot, LLM, agent, embeddings. Primary.
- *Strict GenAI-only* — LLM, GPT, Claude, generative AI, agentic, RAG, MCP, prompt engineering, GitHub Copilot. Sensitivity.

**Other taxonomy lists named with the same care, all in appendix verbatim:**

- People-management markers (direct report, team lead, supervise / oversee, 1-on-1, headcount, performance review).
- Scope / orchestration / mentorship markers.
- Process-scaffolding markers (agile, scrum, requirements, V&V, specification, coordinate, schedule).
- Legacy-stack markers (.NET, COBOL, mainframe, VMware, Active Directory).
- Tech-word lexicon (135 techs, tokenisation calibrated for fairness across older and newer terms).

**Why explicit.** A regex is a measurement tool, not a definition. Reviewers will ask "what's in the list," and the answer should be visible.

### 2.6 Seniority definitions

**Question.** How do we cut junior vs senior?

- **Considered:** direct LLM (`seniority_final`); YOE-based (J3 / S4); intersection of direct and YOE; title-based heuristics (deprecated).
- **Chose:** direct extraction primary; YOE-based and intersection reported as ablations in every junior/senior-dependent claim.
- **Why both:** the two extractors disagree on the *junior-floor* direction (rule shows rise; LLM shows fall). Until a hand-labelled validation sample resolves the disagreement, the headline claim is the *explicit "junior / entry-level" textual marker*, which is robust across rubrics. YOE-floor change is reported as suggestive pending audit.

### 2.7 What JDs are and aren't (scoping discipline)

- JDs are *signals* employers send: about the role, about firm AI-readiness, about what they screen on, about what they signal to investors and the public.
- JDs are *not* employment, hires, realised demand, or salary outcomes.
- Every headline claim is in the register of the *filter*, not the work.
- Causal claims to payroll outcomes are out of scope; the publicly available payroll evidence on AI-occupation employment is contested across studies and we do not attempt the bridge.

### 2.8 Qualitative methodology

**Sample.** 12–15 senior software engineers, US-based, stratified across firm size (FAANG / mid-cap / startup) and industry (tech / public-finance / defense / aggregator-mediated).

**Recruiting.** Snowball + LinkedIn outreach. Target 12+ for saturation-claim defensibility per AIES qualitative norms.

**Protocol.** Semi-structured, theory-driven probe set mapping to (a) the four quantitative margins — role emergence/shrinkage, within-role mutation, seniority redistribution, signal redesign — and (b) the four harm populations identified in §Discussion.

**Recording and pseudonymisation.** Recorded with consent; transcribed with accuracy check; pseudonymised before analysis.

**Analysis.** Reflexive thematic analysis. Two authors independently open-code an initial subset; develop preliminary codebook against the four-margin / four-harm theory frame; code full dataset; iteratively refine. Negative / disconfirming cases checked. LLMs used for transcript familiarisation and excerpt retrieval; *all coding decisions, themes, and quoted material reviewed by authors.*

**Saturation criterion.** No new themes in the last 2–3 interviews; reported explicitly.

**Inter-rater reliability.** Not computed (reflexive thematic analysis discourages it; consistency claim runs through codebook auditability + adjudication notes).

**Released.** Codebook, anonymised excerpts. Full transcripts withheld.

---

## Reading these in slide order

If you read just the **Question / Chose / Why** lines end-to-end, the slide deck for the methods half of the talk is already there. Each `### N.M` block is one slide. Numbers and citations stay attached to the bullets they justify, so individual blocks copy without losing provenance.
