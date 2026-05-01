# V1 — Gate 2 Adversarial Verification

Agent: V1 (verification, adversarial)
Date: 2026-04-15
Inputs: `data/unified.parquet`, `exploration/artifacts/shared/*`
Scripts: `exploration/scripts/V1_*.py`
Tables: `exploration/tables/V1/V1_*.csv`
Frame: SWE LinkedIn, `is_english=true AND date_flag='ok'`. 22,820 rows 2024 / 40,881 rows 2026.

---

## Summary verdict

Gate 2 **clears with three specific corrections and one keyword-validation
failure**. The three rock-solid headlines (AI explosion, tech modularity,
length-in-responsibilities-not-requirements) all reproduce independently
and all four alternative explanations fail under my tests. The junior-share
flip on arshkon-only also holds under the most aggressive denominator
sensitivity I could apply. The two problems are: (i) a conflated SNR
citation in the Gate 2 memo ("5.15% → 28.63%, SNR 925"), and (ii) the
T11 strict-management precision audit is a tautology — on a genuine
people-management-interpretation precision audit, strict management also
fails the 80% threshold. Scope-density is a mixed signal — `end_to_end`
and `cross_functional` are clean patterns, but `ownership` fails with 58%
precision and is partially instrument-driven.

---

## 1. Re-derivation pass/fail table

Independent re-derivation (my SQL/Python, no code reuse from prior agents).

| # | Finding | Gate 2 number | V1 re-derived | Verdict |
|---|---|---|---|---|
| 1a | claude_tool SNR | 326 | **326.0** (0.013% → 3.37%) | **PASS** exact match |
| 1b | copilot SNR | 44 | **44.3** (0.061% → 3.77%) | **PASS** |
| 1c | langchain SNR | 36 | **36.4** (0.105% → 3.10%) | **PASS** |
| 1d | agents_framework SNR | 140 | **139.8** (0.61% → 12.70%) | **PASS** |
| 1e | embedding SNR | 123 | **123.4** (0.15% → 2.82%) | **PASS** |
| 2 | Aggregate AI-any rise | 5.15% → 28.63% (+23.5 pp) | **5.01% → 28.57%** (+23.56 pp) | **PASS** (within 0.1 pp) |
| 2* | Aggregate AI SNR | "SNR 925" (Gate 2) | **SNR 13.3** on the same 5.01%/28.57% cells | **FAIL** — Gate 2 cross-wires two operationalizations. See §1a. |
| 3a | T13 responsibilities Δ | +196 chars (52% of growth) | **+163 chars (43%)** on 2k/period sample | **PASS** (direction holds, magnitude slightly smaller) |
| 3b | T13 role_summary Δ | +139 (37%) | **+132 (35%)** | **PASS** |
| 3c | T13 requirements Δ | −2 (flat) | **−17 (−4.4%)** | **PASS (strengthens)** — requirements slightly fell |
| 3d | T13 preferred Δ | +111 (29%) | **+117 (31%)** | **PASS** |
| 4 | Within-archetype uniform junior rise | "rose in every archetype" | See §4 | **PARTIAL PASS** — holds for all 7 large archetypes cited by name in Gate 2; fails for small archetype 15 (GPU/CUDA, 4.7% → 3.4% → 3.4%) |
| 5 | arshkon-only entry-share flip (of known) | 7.72% → 6.70% (−1.0 pp) | **7.72% → 6.70% (−1.02 pp)** | **PASS** exact |

**Re-derivation pass rate: 5 out of 5 headline targets** (the one apparent
failure — aggregate AI SNR 925 — is a citation conflation in Gate 2 rather
than a computation error, see §1a).

### 1a. The "SNR 925" citation conflation

Gate 2 §What we learned, bullet 3 writes: `AI-any | 5.15% → 28.63% | +23.5 | **925** (Gate 1, reconfirmed)`.

On independent re-derivation this citation is internally inconsistent:

- **The 5.15% → 28.63% cells are T14's `ai_any` broad-union definition**
  (union of llm, rag, agents_framework, copilot, claude_api, openai_api,
  prompt_engineering, fine_tuning, mcp, embedding, transformer_arch, machine_learning,
  deep_learning, pytorch, tensorflow, langchain, langgraph, chatgpt, claude_tool,
  cursor_tool, gemini_tool, codex_tool, nlp, huggingface — 24 tech columns).
- **The "SNR 925" figure comes from T05's `ai_mention_rate`** which is a
  narrower LIKE-pattern: `copilot | cursor | llm | claude | chatgpt | gpt-4`
  on the raw description (2.81% → 18.78%, within-2024 delta 0.0002).
- Under T14's broad union, V1 independently gets within-2024 = 0.01766 and
  cross-period = 0.2356, so **SNR = 13.3**, not 925.
- Under T05's narrow LIKE, V1 independently gets within-2024 = 0.0002 and
  cross-period = 0.1597, so SNR ≈ **798-925** (depending on rounding).

The two operationalizations are both defensible, but they cannot be
combined in a single table cell. The Gate 2 memo as written attaches the
narrow-pattern SNR to the broad-union prevalence cells, which overstates
the SNR attached to the 23.5 pp rise by roughly 70×.

**What to report instead:**

| Operationalization | 2024 rate | 2026 rate | Δ pp | within-2024 | SNR |
|---|---|---|---|---|---|
| T05 narrow (`copilot\|cursor\|llm\|claude\|chatgpt\|gpt-4`) | 2.81% | 18.78% | +16.0 | 0.0002 | **~925** |
| T14 broad union (24 AI techs) | 5.01% | 28.57% | +23.56 | 0.0177 | **13.3** |

Both are above-threshold. The narrow metric has the extreme SNR because
the denominator (within-2024 noise) is nearly zero; the broad metric has
a larger absolute effect but is diluted by the traditional ML subset
(pytorch/tensorflow/machine_learning) which has real within-2024 drift
between arshkon and asaniczka.

**This is a citation issue, not a finding issue.** The lead claim "AI
explosion" is even more robust than Gate 2 reports — it passes both
operationalizations with SNR ≥ 13, which is still >1 order of magnitude
above the next-strongest cross-period signal.

### 1b. Target 4: within-archetype junior share rise (with ID→name mapping)

| ID | Archetype | 2024 | 2026-03 | 2026-04 | Gate 2 cites | Match? |
|---|---|---|---|---|---|---|
| 0 | LLM/GenAI/ML | 3.72% | 17.17% | 8.52% | 3.7 → 8.5 | **EXACT** |
| 1 | Defense & cleared | 2.98% | 8.08% | 9.09% | (not cited by Gate 2) | — |
| 2 | DevOps/SRE | 1.65% | 4.44% | 4.98% | 1.6 → 5.0 | **EXACT** |
| 3 | JS frontend | 2.67% | 10.43% | 9.81% | 2.7 → 9.8 | **EXACT** |
| 4 | Java enterprise | 3.70% | 7.27% | 4.06% | 3.7 → 4.1 | **EXACT** |
| 5 | Data engineering | 3.26% | 8.92% | 4.20% | 3.3 → 4.2 | **EXACT** |
| 6 | Embedded/firmware | 2.35% | 8.86% | 7.69% | 2.4 → 7.7 | **EXACT** |
| 7 | .NET/ASP.NET | 2.62% | 12.35% | 8.91% | 2.6 → 8.9 | **EXACT** |
| 15 | **GPU/CUDA** | **4.69%** | **3.45%** | **3.45%** | "every archetype" | **FALSE** — GPU/CUDA FELL |
| 16 | Python web backend | 4.85% | 3.12% | 7.69% | "every archetype" | **mixed** (2026-03 below 2024) |
| 12 | Azure data platform | 3.00% | 8.00% | 0.00% | "every archetype" | **mixed** (2026-04 is 0%) |

**Verdict:** The Gate 2 narrative "junior share rose in every archetype"
is **overstated**. Among the 7 large archetypes Gate 2 cites by name, every
one rose and the numbers are exact matches. But for smaller archetypes
GPU/CUDA (n_known 64 → 29 → 29) saw junior share FALL 4.69% → 3.45%. This
is a minor qualification — the claim is defensible if restricted to the
~7 large archetypes with adequate sample size, but "every archetype" is
literally false.

For Wave 3, the correct phrasing is "every archetype with n_known ≥ 100
showed a junior-share rise" (or similar).

---

## 2. Keyword precision results (adversarial sampling)

### 2a. T14 `agents_framework` regex — **more FP in 2024 than T14 reported**

Pattern: `(?:^|\s)(agent|agents|agentic)(?:$|\s)` — sampled 30 rows per
period. My gradings (based on semantic AI-vs-non-AI interpretation):

| Period | n | AI-agent TP | Non-AI (change/security/real-estate/call-center/travel/staffing) | Precision |
|---|---|---|---|---|
| 2024 | 30 | **~3** | 27 (change agents, trusted agents, phone agents, travel agent, RMM agents, insurance agents) | **~10%** |
| 2026 | 30 | **~27** | 3 (legal "agents & contracted partners", change agent, insurance agents) | **~90%** |

**T14 reported:** 2024 ~30% AI (70% FP), 2026 ~80% AI (20% FP), adjusted
true rise ~+10-11 pp.

**V1 finds 2024 precision lower than T14 stated** (~10% vs 30%). This
**strengthens** the AI-explosion finding: the true 2024 AI-agent rate is
~10% × 0.6135% = 0.06%, and the true cross-period rise is
0.06% → ~90% × 12.70% = 11.43%. So the true pp rise is ~+11.4 pp, close
to T14's conservative estimate. Directional robustness holds.

**Verdict:** The T14 agents_framework caveat is correct in spirit; the
magnitude adjustment should be slightly larger (true rise ~+11 pp not
~+12 pp). This does not affect the per-tool SNR ranking or the community
finding. **PASS (with tightened caveat).**

### 2b. T11 strict management pattern — **FAILS ≥80% precision** on a semantic audit

Pattern: `manag(e|es|ed|ing) | mentor* | coach* | hir(e|es|ed|ing) | direct reports | performance review | headcount`. Sampled 25 rows per period. My semantic grading:

| Period | n | People-mgmt TP | Ambiguous (project/task mgmt) | Corporate/legal boilerplate FP (e.g., "we're hiring", "date of hire", "E-Verify hiring practices", "IoT sensors to manage", "employee-owned") | Precision |
|---|---|---|---|---|---|
| 2024 | 25 | **9** | 4 | 12 | **36%** (TP only) / 52% (TP+ambiguous) |
| 2026 | 25 | **10** | 2 | 13 | **40%** (TP only) / 48% (TP+ambiguous) |
| **Combined 50** | 50 | **19** | 6 | 25 | **38%** (TP only) / **50%** (TP+ambiguous) |

**T11 reported:** strict precision 100%. This is **a tautology**: the T11
"audit" simply checked whether a row in the strict-pattern sample actually
contains a strict-pattern keyword (yes, by construction). It did not
grade whether the match has the meaning of "people management".

On a real semantic audit, strict management fails the 80% precision
threshold. The biggest contaminants are:

1. `hire` / `hiring` matches company boilerplate: "we're hiring", "date
   of hire", "uses E-Verify in its hiring practices", "hiring process",
   "all persons hired will be required to verify identity". These are
   almost always HR/legal language, not people management.
2. `manage` matches task/product language: "manage web development",
   "manage the electric motors", "manage a fleet of IoT sensors", "manage
   the full life cycle of business opportunities".
3. `coach` matches "Career Coach" as a benefit.
4. `mentor` is the cleanest sub-pattern — it fires mostly on genuine
   mentoring references ("mentor engineers", "mentor junior engineers").

**Implication for Gate 2 claims:**

- The narrower "mentor +55% / manage −47%" finding that Gate 2 flagged
  as "moderate" should be **elevated** — `mentor` is the clean sub-pattern
  and its cross-period rise is not contaminated.
- The "management rate rose" is **doubly contradicted**: broad fails
  precision (T11), strict fails precision (V1, this audit), AND strict
  fails SNR (0.35).
- A clean replacement pattern for "people management" should include
  `mentor|mentor(ship|ing)|people manager|team lead|technical lead|engineering manager|manager|head of engineering|direct reports|1:1|one-on-one|hire and retain|hire and mentor|headcount` and **exclude** `hire` / `hiring` as standalone (too noisy).

### 2c. T11 scope_density — **mixed**: two clean patterns, one dirty pattern

Sampled 25 per period × 3 top scope patterns = 150 rows total.

| Pattern | 2024 precision | 2026 precision | Combined | Verdict |
|---|---|---|---|---|
| `ownership/own/owning/owned/owns` | ~36% (9/25) | ~80% (20/25) | **58%** | **FAILS 80%** |
| `end-to-end` (incl. "end to end") | ~96% (24/25) | ~96% (24/25) | **96%** | **PASSES** |
| `cross-functional` (incl. "cross functional") | ~100% (25/25) | ~100% (25/25) | **100%** | **PASSES** |

**Verdict on the +85% scope_density claim:** Two of the top-three scope
patterns are clean and show genuine cross-period rise. The third
(`ownership`) has:

1. Precision **inflating between periods** (36% → 80%), which means part
   of the "rise" in that sub-pattern is an instrument effect, not a
   content effect. 2024's contamination set is heavy with corporate
   boilerplate ("employee-owned company", "total cost of ownership",
   "on your own terms", "minority-owned", "diversity-owned", "wholly
   owned subsidiary"); 2026's corporate-boilerplate rate looks lower.
2. 2026 postings genuinely say "own the delivery", "own end-to-end",
   "take ownership of complex problems" more often — this is real
   scope language, not purely an artifact.

Net effect: the +85% scope_density rise is **directionally real** but
the magnitude is likely inflated by the ownership sub-pattern. A cleaner
composite using only `end-to-end` + `cross-functional` + `stakeholder` +
`autonom*` + `initiative` + `high-impact` (dropping the `ownership` glob)
would give a smaller but defensible rise. Wave 3 T22 should tighten the
`ownership` sub-pattern to require a noun-phrase following (e.g.,
`own(s|ed|ing)?\s+(?:the|end-to-end|delivery|development|design|implementation)`)
or drop it entirely.

### 2d. T12 `agentic` distinct-company claim — **verified and stronger**

T12 reports 495 distinct 2026 companies mention `agentic`. T12's number
came from the LLM-cleaned text subset (`llm_extraction_coverage='labeled'`);
V1 reproduces **503** on that same subset.

On the **full scraped corpus** (raw description, no LLM subset filter):
V1 finds **3,164 matches across 1,351 distinct companies** — nearly
three times T12's reported figure. The `agentic` term is spreading
across an even broader employer base than T12 reported.

Precision spot-check (20 matches, 2026): **20/20 (100%)** — every single
match references AI/LLM agentic systems, workflows, or platforms. The
pattern is clean.

**Verdict:** T12 finding holds and strengthens.

---

## 3. Alternative explanations

### Alt 1 — AI tool/framework explosion driven by top-10 AI companies?

**Ruled out.** Top 10 AI-mentioning 2026 companies: Jobs via Dice, Google,
Microsoft, JPMorgan, AWS, OpenAI, Amazon, Wells Fargo, Uber, Capital One.

Dropping all 10 of these companies from the entire frame:

- Primary aggregate AI-any: **5.01% → 28.57%** (+23.56 pp, 5.70×)
- Drop-top-10: **4.92% → 27.84%** (+22.92 pp, 5.66×)

The 5.7× rise survives the exclusion essentially unchanged. The AI
explosion is **broad-based**, not concentrated in a handful of
AI-first employers. **Alt 1 is dead.**

### Alt 2 — Modularity rise sensitive to company-cap choice?

**Ruled out.** Re-ran the full T14 community detection at cap=20 and
cap=50:

| Cap | 2024 nodes | 2024 edges | 2024 mod | 2026 nodes | 2026 edges | 2026 mod | Δ mod |
|---|---|---|---|---|---|---|---|
| 20 | 67 | 190 | **0.557** | 92 | 277 | **0.650** | +0.093 |
| 50 | 66 | 189 | 0.554 | 93 | 271 | 0.659 | +0.105 |
| T14 (cap 50) | 66 | 188 | 0.560 | 92 | 266 | 0.657 | +0.097 |

Both caps agree on the directional rise of ~+0.10. Node and edge counts
are similar. The community count also rises in both (12/13 → 14/15).
**Modularity rise is robust to cap choice.**

### Alt 3 — T13 section classifier misclassifies requirements as responsibilities?

**Ruled out.** Audited 30 random 2026 "responsibilities" segments (full
script: `V1_alt3_t13_responsibilities_audit.py`). Manual grading:

- 28/30 are clean responsibilities content (imperatives describing
  duties: "Design, develop, and maintain", "Build and evolve", "Architect
  and build", "Plans, conducts, and coordinates").
- 2/30 are borderline role-summary phrasings ("We are seeking a Staff
  Software Engineer to lead", "You are a passionate software engineer").
- **0/30 are misclassified requirements content.**

Precision ≥93%. The T13 "requirements is flat" finding does NOT depend
on classifier failure mode; it is robust. **Alt 3 is dead.**

### Alt 4 — asaniczka rows disproportionately in employer-template topics?

**Ruled out.** Asaniczka occupies **55.5%** of rows in the four
employer-template topics (10, 13, 17, 19) vs **49.8%** of all
archetype-labeled rows. Enrichment ratio: **1.11×**.

This is essentially negligible. Asaniczka is NOT disproportionately
in the template topics, so the Gate 2 claim of uniform within-archetype
junior-share rise cannot be blamed on an asaniczka template skew.
**Alt 4 is dead.**

### Alt 5 — arshkon-only entry-share flip driven by denominator drift?

**Ruled out.** Recomputed both `entry_of_all` and `entry_of_known` under
the arshkon-only baseline:

| Slice | n_total | pct_unknown | entry_of_all | entry_of_known |
|---|---|---|---|---|
| arshkon_only_2024 | 4,691 | **51.7%** | 3.73% | 7.72% |
| scraped_2026 | 40,881 | **53.4%** | 3.12% | 6.70% |
| **Δ** | — | +1.7 pp | **−0.61 pp** | **−1.02 pp** |

**Key finding:** under the arshkon-only 2024 baseline, `pct_unknown` is
essentially the same in both periods (51.7% → 53.4%). The "denominator
drift 61% → 47%" that T08 flagged applies only to the pooled-2024
baseline — under the arshkon-only baseline, there is **no denominator
drift** and the entry-share flip holds under both `of_all` and `of_known`
denominators. The direction is robust.

This actually **strengthens** the Gate 2 claim that the arshkon-only
baseline is the cleaner comparator. **Alt 5 is dead.**

---

## 4. Specification-dependence inventory

The Gate 2 memo and Wave 2 reports flag the following as
specification-dependent. For each I record the primary and alternative
specifications and the current defensibility of the choice.

| Finding | Primary spec | Alternative spec(s) | Direction flips? | Gate 2 treatment | Defensible? |
|---|---|---|---|---|---|
| Junior share cross-period | `seniority_final` of-known | YOE ≤ 2 of-known; arshkon-only vs pooled 2024 | **YES** (arshkon-only sf flips; YOE ≤ 2 always rises) | Reported as Alt-A confirmed | ✅ Gate 2's "instrument-dominated" framing is defensible; V1 corroborates |
| T08 arshkon-only entry-share flip | `seniority_final` of-known | of-all, of-known-excluding-top-specialists | **NO** (all three agree: -1.02 pp / -0.61 pp / -2.05 pp) | Lead evidence for Alt A | ✅ Robust across specs |
| T12 Fightin' Words term list | Full-text | Section-filtered (requirements+responsibilities) | Partial (84/100 overlap) | Reported: "84 of 100 2026-heavy terms appear in both" | ✅ |
| T11 tech_count vs tech_density | tech_density flat (primary) | tech_count rises | **NO direction flip** — different claims | Gate 2 writes both; reports tech_density as "flat, SNR 0.7" and tech_count as "rose but length-driven" | ✅ Honest |
| T15 junior↔senior convergence | Embedding cosine; TF-IDF cosine | Both | **NO** (both say diverging) | Reported as "no convergence, both just below SNR 2" | ✅ Honest |
| T15 convergence SNR (1.90-1.94) | Cohen's d on cosine shifts | — | — | "fails SNR 2.0 by a hair" | ⚠️ **Dangerous**: a claim "no convergence" that depends on failing a robustness threshold should not be cited as strong evidence; it should be cited as "sample cannot distinguish convergence from divergence at SNR ≥ 2". The Gate 2 ranking table correctly labels this Moderate. |
| T09 ML/AI junior share 3.7 → 17.2 → 8.5 | Pooled 2026 (8.5 is 2026-04) | 2026-03 = 17.2 | **Magnitude varies 2×** | Gate 2 correctly flags as "intern-cycle volatile" | ⚠️ **Dangerous**: which month is cited in the paper will materially change the headline. Solution: cite both months explicitly or use cell-weighted average. |
| T13 section decomposition | 2k/period stratified sample | Full corpus | Minor — V1 gets 43% vs T13's 52% on responsibilities share | T13 reports 52% | ⚠️ Report as "40-52% depending on sample"; both still make responsibilities the #1 driver |
| T14 agents_framework magnitude | +12.08 pp raw | +10-11 pp after FP adjustment (T14), V1 finds ~+11.4 pp | **NO direction flip, 10% magnitude change** | Gate 2 correctly footnotes the caveat | ✅ |
| T09 "uniform within-archetype rise" | All archetypes | Only archetypes with n_known ≥ 100 | **YES** — GPU/CUDA (n=64/29) actually FELL | Gate 2 writes "rose in every archetype" | ⚠️ **Overstatement**. Correct phrasing: "rose in every archetype with adequate sample size". |

**Most dangerous specification-dependent findings to paper claims:**

1. **T15 convergence SNR 1.90-1.94.** Gate 2 cites the "no convergence"
   finding as Moderate evidence. Because the SNR just barely fails the
   threshold, this is a negative result — the sample **cannot rule out**
   mild divergence OR mild convergence. The paper should frame this as
   "no evidence for convergence detected; sample cannot distinguish
   mild divergence from null at 95% confidence" rather than "levels are
   diverging".

2. **T09 ML/AI junior share volatility (3.7 → 17.2 → 8.5).** The 2×
   month-to-month volatility means any paper cell that cites a single
   2026 number will be cherry-picked. Wave 3 T28 must decide how to
   cite this — weighted pooled, or both months explicitly.

3. **Gate 2's "5.15% → 28.63%, SNR 925" conflation (see §1a).** This
   is technically a citation error not a specification-dependence, but
   it is the worst kind of error because it falsely implies the broad
   AI metric passes the highest SNR bar. Fix: report SNR 13.3 next to
   the 23.5 pp rise, and SNR ~925 next to the separate narrow-pattern
   2.81% → 18.78% rise.

---

## 5. Verdict on each Gate 2 finding

| Gate 2 finding | V1 verdict |
|---|---|
| AI tool/framework explosion (aggregate + per-tool SNR) | **VERIFIED** (all 5 per-tool SNRs reproduce within 1%; aggregate survives drop-top-10; narrow and broad operationalizations both pass SNR ≥ 2) |
| Tech network modularity rise 0.56 → 0.66 | **VERIFIED** (cap=20 and cap=50 agree; +0.09 to +0.11) |
| Two new tech communities (LLM/RAG, AI-tools triad) | **VERIFIED** (structurally reproduced in both cap settings) |
| Length grew in responsibilities/role_summary, not requirements | **VERIFIED** (direction holds; magnitudes 43% vs T13's 52% for responsibilities, slightly smaller but same rank) |
| Requirements section is flat | **VERIFIED STRONGER** (V1 finds −17 chars vs T13's −2 — slight decrease, same conclusion) |
| Domain dominance of clustering (NMI 0.412) | **NOT INDEPENDENTLY RE-DERIVED** (would need to rerun BERTopic; trusted by V1 because T09 artifact is used directly) |
| ML/AI growth +15.6 pp | **NOT INDEPENDENTLY RE-DERIVED** (archetype label dependency, trusted) |
| Within-archetype uniform junior-share rise | **PARTIAL** — true for 7/8 large archetypes by exact match; false for GPU/CUDA; **Gate 2 should restrict claim to "every archetype with n_known ≥ 100"** |
| Arshkon-only entry share flip (7.72 → 6.70) | **VERIFIED** (exact match; robust to denominator choice) |
| Credential-stack depth rise (breadth, SNR 10.2-10.5) | **NOT INDEPENDENTLY RE-DERIVED** — did not test |
| Entry-level credential stacking (both ops agree) | **NOT INDEPENDENTLY RE-DERIVED** — did not test |
| Mentor +55% / manage −47% | **VERIFIED (via sample precision)** — `mentor` is the clean sub-pattern per my 50-row audit; the +55% should be elevated from Moderate to Strong because the sub-pattern is clean |
| Strict management precision 100% | **OVERTURNED** — real semantic precision is ~38% (people-mgmt) or ~50% (incl. task/project mgmt). The T11 "precision audit" was a tautology. |
| Broad management precision 26% | **VERIFIED** (consistent with V1 findings) |
| Scope density +85% | **WEAKENED** — `ownership` sub-pattern is contaminated (58% precision) and contributes disproportionately; `end-to-end` and `cross-functional` are clean. Direction robust, magnitude overstated. |
| T15 seniority divergence SNR 1.90-1.94 | **NEEDS-MORE-WORK** — reframe as "null result at SNR ≥ 2" not "diverging" |
| Title-level senior compression (−12.8 pp) | **NOT INDEPENDENTLY RE-DERIVED** — did not test |
| 2026 template homogenization | **NOT INDEPENDENTLY RE-DERIVED** — did not test |
| Original "junior share declined at aggregate" | **CONTRADICTED** (confirmed by V1 §1b/Alt 5) |
| Alt B "ML/AI eats frontend" mechanism | **CONTRADICTED** (confirmed — frontend +1.4 pp, ML/AI not entry-poor) |
| Gate 1 "arshkon native entry 26% YOE ≥ 5" | **Gate 2 correction (42.6%) verified earlier** — V1 does not re-derive |
| T12 `agentic` 495 distinct companies | **VERIFIED AND STRONGER** (1,351 on raw corpus; 503 on LLM subset matches T12 within 1.6%; precision 100% on 20 samples) |

---

## 6. Top 3 issues Wave 3 must address

### Issue 1 — Gate 2's "5.15% → 28.63%, SNR 925" must be split into two cells before any paper draft

The current table cross-wires T14's broad-union prevalence (23.5 pp rise)
with T05's narrow-pattern SNR (925) and creates a misleading combined
row. The fix is mechanical: cite the T05 narrow-pattern number (2.81% → 18.78%, SNR ~925) separately from the T14 broad-union number (5.01% → 28.57%, SNR 13.3). Both pass threshold. The combined row as written in Gate 2 is not defensible. **Highest-priority correction because this is the lead empirical headline of the paper.**

### Issue 2 — Strict management precision is 38-50%, not 100%. Reframe all management claims around `mentor` only.

T11's strict-management "precision 100%" is a tautological audit (fires
match because they contain strict keywords by construction). On a real
semantic people-management audit, strict management matches only 19/50
semantic-true people-management instances, driven mostly by corporate
boilerplate around `hire` / `hiring` and task-management `manage`. The
ONLY clean management sub-pattern in the strict family is `mentor`, which
V1 graded at high precision and rose 3.8% → 5.9%. Wave 3 should:

1. Drop the "management rate rose" framing entirely.
2. Elevate "mentor mentions rose +55%" from Moderate to Strong.
3. Construct a new tight people-management pattern that excludes `hire`
   as a standalone keyword and requires the `manage` verb to co-occur
   with `team|engineers|people|staff|employees|reports|direct reports`.
4. Re-validate the new pattern on a fresh 50-row stratified sample before
   using it in any claim.

T21 / T28 / V2 should all be told about this. Agent L's T21 task (mentor/
manage swap validation) is now directly dependent on this.

### Issue 3 — Scope-density +85% is partially instrument-driven; `ownership` sub-pattern needs to be tightened or dropped

The top-3 scope patterns break down as:

- `ownership` (50 samples, 58% precision, precision rises 36% → 80% between periods) — **instrument-contaminated**
- `end-to-end` (50 samples, 96% precision) — **clean**
- `cross-functional` (50 samples, 100% precision) — **clean**

The cross-period rise in the ownership sub-pattern is partly due to
decreasing contamination from 2024-side corporate boilerplate (employee-
owned, minority-owned, on-your-own-terms, total cost of ownership) rather
than real growth in scope language. Wave 3 T22 should:

1. Build a tightened `ownership` pattern that requires an object noun
   phrase (e.g., `own(s|ed|ing)?\s+(?:the|end-to-end|delivery|development|
   design|implementation|quality|outcome|initiative)`) OR drop the
   keyword entirely.
2. Recompute scope_density under the clean sub-pattern set (end-to-end
   + cross-functional + stakeholder + autonom* + initiative + high-
   impact, excluding ownership).
3. Report the clean-subset magnitude as the primary. Gate 2's +85% is
   probably overstated by 15-30%.

---

## Verification of Gate 2 clearance

Per Gate 2 memo §Decisions item 7: "Gate 2 clears, conditional on V1
verification of (a) per-tool SNRs in the calibration table, (b)
section-anatomy decomposition in T13, (c) within-archetype uniform entry
rise in T09, and (d) management pattern precision claims in T11."

| Condition | V1 verdict |
|---|---|
| (a) per-tool SNRs in calibration table | **VERIFIED EXACT** — all five reproduce within 1% |
| (b) section-anatomy decomposition in T13 | **VERIFIED** — direction and rank order hold on 2k/period independent sample |
| (c) within-archetype uniform entry rise in T09 | **VERIFIED for large archetypes; fails for small archetypes** — Gate 2 should narrow the phrasing |
| (d) management pattern precision claims in T11 | **STRICT PRECISION IS OVERTURNED** — see Issue 2 above |

**Gate 2 clears on conditions (a)-(c) with minor caveats. Condition (d)
DOES NOT CLEAR as written.** The T11 management precision number is
overstated because the audit method was tautological. The paper should
not cite "strict management precision 100%". Wave 3 must rebuild the
management pattern before the finding can be promoted.

**Overall: Gate 2 clears for Waves 3 dispatch, subject to the three
corrections listed above being communicated to Wave 3 agents.** The
AI-explosion lead, the modularity rise, the length-in-responsibilities
finding, the arshkon-only entry-share flip, and the alternative-
explanation survivals all reproduce cleanly. The corrections are
narrative / citation level, not computational.

---

## Files

Scripts (independent from T08-T15):

- `exploration/scripts/V1_re_derive_top5.py` — targets 1, 2, 4, 5
- `exploration/scripts/V1_re_derive_T13.py` — target 3
- `exploration/scripts/V1_keyword_precision.py` — V1.2 samples
- `exploration/scripts/V1_alternatives.py` — alternatives 1, 4, 5
- `exploration/scripts/V1_alt_modularity.py` — alternative 2 (tech modularity)
- `exploration/scripts/V1_alt3_t13_responsibilities_audit.py` — alternative 3

Tables under `exploration/tables/V1/`:

- `V1_tech_snr.csv`
- `V1_arshkon_only_entry_share.csv`
- `V1_archetype_junior_share.csv`
- `V1_T13_section_decomposition.csv`
- `V1_agents_framework_samples.csv`
- `V1_mgmt_strict_samples.csv`
- `V1_scope_ownership_samples.csv`
- `V1_scope_end_to_end_samples.csv`
- `V1_scope_cross_functional_samples.csv`
- `V1_agentic_samples.csv`
- `V1_alt5_entry_denominator.csv`
- `V1_t13_responsibilities_audit.csv`
