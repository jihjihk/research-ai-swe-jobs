# Gate 0 — Pre-Exploration Note

Date: 2026-04-15
Author: Research advisor (orchestrator)

This is a pre-exploration note written before any sub-agent has run, so it can be compared against the real evidence later. The goal is to make my priors explicit so the gates can check them.

## What I know going in

- Input: `data/unified.parquet` (~7 GB), three sources (`kaggle_arshkon`, `kaggle_asaniczka`, `scraped`), LinkedIn-primary, Indeed as sensitivity.
- The canonical seniority column is `seniority_final`, combining Stage-5 strong rules and Stage-10 LLM. `seniority_final_source` says which half fired.
- The only cleaned-text column is `description_core_llm`. Raw `description` is only acceptable for boilerplate-insensitive checks.
- asaniczka has **zero native entry-level labels** — any `seniority_native` sanity check must be arshkon-only.
- LLM budget caps coverage; `selected_for_llm_frame` is the balanced core, `llm_extraction_coverage` / `llm_classification_coverage` are the gates.
- The initial narrative (RQ1-RQ4) is a hypothesis: junior scope inflation, senior archetype shift toward AI orchestration, employer-usage divergence. The advisor brief is explicit that this should be challenged, not defended.

## What I am most confident about

1. **The data exists and is rich enough to show *something* at the aggregate level.** Three sources, longitudinal, ~millions of rows per the preprocessing layout. Some signal will show up regardless of whether the initial narrative is right.
2. **Description length grew substantially from 2024 → 2026.** Multiple docs reference a 56% growth. This is almost certainly a real observation; the open question is whether it's content growth or boilerplate growth.
3. **Aggregators and a small number of prolific employers will dominate naive corpus aggregates.** The T06 preamble spells out a prior incident where ~23% of the 2026 entry pool came from six companies. Concentration is a first-class feature, not a footnote.
4. **Text-based metrics need length normalization and company-name stripping.** Without them the top-distinguishing-term lists will just be "Amazon, Google, Meta, Microsoft" and "benefits, PTO, 401k".

## What I am least confident about, and what would change my mind

### 1. Is "junior scope inflation" real, or is it an artifact?

**Prior:** Weakly supportive, mostly because prior runs have partially seen it. But the artifact risks are serious:
- The 56% length growth means raw keyword counts for scope/management language will grow even if per-1K-char density is flat.
- Differential native-label quality across 2024 sources could mean "entry" in arshkon and "entry" in the Stage-10 LLM for asaniczka/scraped do not measure the same thing.
- The YOE-based proxy (`yoe_extracted <= 2` share) is label-independent and should be the tiebreaker.

**Would move me toward "artifact":** if T13 shows the length growth is mostly in benefits/legal/about-company sections; if `seniority_final` and the YOE proxy disagree in direction; if the within-2024 calibration (arshkon vs asaniczka on the same metric) is larger than the 2024→2026 change.

**Would move me toward "real":** if `tech_density` (per 1K chars) and `scope_density` both rise in the requirements section, if the finding survives company capping, and if the YOE proxy directionally agrees.

### 2. Is the senior archetype shift toward AI/orchestration real, or a general template-shift?

**Prior:** Weakly supportive, but Wave 2 and Wave 3 tasks are explicitly primed to look for "management language expanded at ALL levels" (field-wide template shift) vs. "management language dropped at senior and rose at entry" (redefinition). These are empirically distinguishable. I have low confidence which way it will land.

**Would move me toward "real":** concurrent rise of orchestration/review/agent language within senior postings AND a within-company trajectory in the T16 overlap panel.

**Would move me toward "template shift only":** if generic management language (`lead`, `strategic`, `leadership`) grows at every level and precision validation on 50-sample manual check drops below 0.8.

### 3. Is scope inflation within-company or between-company (composition)?

**Prior:** Genuinely uncertain. T06 and T16 are designed to decompose this, and decomposition routinely flips aggregate findings on labor-market data. I would not be surprised to learn that most of the "aggregate entry decline" is between-company (different companies posting), which is a *different paper* than "companies changed what they hire for."

### 4. How much of the apparent 2024→2026 change is instrument noise?

**Prior:** A lot of it, probably. Kaggle is HTML-stripped; scraped is markdown-preserved. The within-2024 (arshkon vs asaniczka) calibration from T05/T08 will tell us the noise floor. If the 2024→2026 effect sizes are less than 2× the within-2024 effect sizes, the finding is indistinguishable from cross-source variation.

### 5. Will T09's archetype clusters organize by seniority, by tech domain, or by company type?

**Prior:** Tech domain. Most unsupervised clustering on job postings in prior labor-market work finds domain dominance (Frontend / Backend / Data / ML / Embedded / …). If this holds, the "junior decline" conversation needs to be re-asked as "did the market recompose across domains, and is the entry-poor AI/ML domain growing?" That's a reframing, not a minor caveat.

## Narrative stance at Gate 0

I am holding the RQ1-RQ4 framing loosely. The most interesting possible outcomes, ranked by how much they would shift the paper:

1. **Domain recomposition explains most of the aggregate junior decline.** The market shifted from domains that historically hired juniors to domains that don't. The paper becomes a "skill-surface restructuring" paper, not a "junior rung narrowing" paper.
2. **LLM-authored descriptions are driving the content change.** T29 is exploratory but the finding would be unifying and paradigm-shifting for the method: we'd be measuring recruiter tool adoption as much as employer restructuring.
3. **Senior archetype shift is the headline, junior story is weak.** The opposite rank-order from RQ1.
4. **Ghost requirements are the headline.** T22 finds that AI requirements are substantially more aspirational than traditional ones, and the paper becomes about "what employers *say* they want" with AI as the accelerant.
5. **Everything survives and RQ1-RQ4 are confirmed.** Possible but I'd flag it as "the most expected outcome" which, in exploration, is usually not the most interesting one.

I am deliberately entering Wave 1 expecting to be surprised on at least one of these.

## Acceptance criteria for Gate 1

For the gate memo to clear, I want:
- Actual row counts by source × period × `is_swe` — not the documented ones.
- `llm_extraction_coverage` and `llm_classification_coverage` by source with `labeled` share reported.
- T02 verdict on whether asaniczka `associate` can serve as a junior proxy.
- T03 three-way entry-share comparison: `seniority_final` vs arshkon-only `seniority_native` vs YOE-based proxy. If they disagree, the mechanism is investigated before Wave 2 builds on any of them.
- T06 concentration table + entry-poster concentration + duplicate-template audit + per-finding concentration prediction table.
- T07 feasibility table stating which cross-period comparisons are well-powered.

If any of these is missing the gate doesn't clear and I re-dispatch.
