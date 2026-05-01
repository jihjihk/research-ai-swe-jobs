# How the story evolved across gates

Four gate reviews, two verification passes, three dominant framings overturned, one research question inverted. This page traces how the narrative reached its current state.

## Gate 0 — Pre-exploration (baseline hypotheses)

**Original research questions:**

- **RQ1:** Is the junior rung narrowing? Are entry-level postings shrinking while mid-senior requirements inflate?
- **RQ2:** Are senior roles being redefined by AI? Are seniority levels blurring?
- **RQ3:** Do employer AI requirements outpace worker adoption? (The popular "employers demand impossible AI skills" framing.)
- **RQ4:** What do qualitative interviews say about these changes?

**Expected story (going in):** junior share falls, seniority blurs as work converges around AI, employers push AI skill demands ahead of worker readiness.

See: [`audit/memos/gate_0_pre_exploration.md`](../audit/memos/gate_0_pre_exploration.md)

## Gate 1 — Wave 1 verdict (data foundation)

**What survived:**

- BLS geographic validation r=0.97 — the sample frame is valid.
- `seniority_final` as the primary seniority column (combined rule + LLM).
- Overlap panel n=240 at ≥3 SWE per period.

**What broke:**

- **`seniority_native` is not usable as a 2024 baseline.** Arshkon native `entry` mean YOE is 4.18 (42.6% at YOE ≥ 5); asaniczka has zero native entry labels.
- **Asaniczka `associate` is NOT a junior proxy** — 88% lands in mid-senior under `seniority_final`.
- **Entry cross-period pooled-2024-vs-scraped is marginal** (MDE 8.2 pp).
- **Only AI mention rate is SNR-robust** under within-2024 calibration.

**RQ implications:** RQ1 junior-narrowing is already in trouble at the foundation level. RQ3 is the best-positioned original RQ.

See: [`audit/memos/gate_1.md`](../audit/memos/gate_1.md)

## Gate 2 — Wave 2 verdict (open structural discovery)

**The explosion is real.** Narrow AI SNR 925. Broad AI SNR 13.3. Tech network modularity rose.

**The first reframing:**

- **Length grew in responsibilities + role_summary, NOT requirements** — interesting, but consistent with either "scope inflation" or "style migration" at this point.
- **Junior-share rise is instrument + composition artifact** — arshkon-only flip, denominator drift, uniform rise across archetypes. Entry 2026 content is genuinely entry-level (not mid-senior relabeled).
- **No seniority convergence** — T15 embedding SNR 1.90/1.94, below threshold. The "roles are converging" framing is a null.
- **Senior title compression** (new finding) — "senior" 41.7% → 28.9%, "staff" +3.1 pp.

**First V1 verification pass (Gate 2 verification):**

- 5/5 headline numbers reproduced within 1%.
- **Three corrections:**
  1. The SNR 925 narrow cell and the +23.5 pp broad cell had been cross-wired. They are from different tasks and metrics. **Cite separately.**
  2. T11's "management pattern 100% precision" was tautological — real semantic precision 38-50%. Only the `mentor` sub-pattern is clean.
  3. T11's `scope_ownership` sub-pattern is 58% precision with 2024-side contamination. `end-to-end` and `cross-functional` are clean.

See: [`audit/memos/gate_2.md`](../audit/memos/gate_2.md), [`gate_2_corrections.md`](../audit/memos/gate_2_corrections.md), [`V1_verification.md`](../audit/verifications/V1_verification.md)

## Gate 3 — Wave 3 verdict (market dynamics)

**The biggest story evolution in the whole exploration.** Several Gate 2 headlines get substantially reframed or overturned.

**T21 rebuilt T11's management patterns from scratch with validated object-noun-phrase regex.** Result:

- "Manage −47%" at mid-senior is **overturned**. Narrow, validated mid-senior people-management actually **rose +25%**.
- **Mid-senior technical orchestration rose +98%, director orchestration +156%, director people-management FELL 21%.** The real story is specialization toward tech orchestration, not general management shift.
- **AI × senior interaction is localized entirely in orchestration** — people-management density is identical regardless of AI. The tech-lead sub-archetype doubled.
- **Staff is NOT the new senior** — staff absorbs only ~22% of the senior-title drop on the overlap panel.

**T18/T19 show the AI explosion is field-wide and 92% within-company.**

- SWE-vs-control DiD +29.6 pp; SWE-vs-adjacent +2.1 pp. **Not SWE-specific.**
- Macro-robustness ratio 24.7× on broad AI. JOLTS cooling is not confounding.

**T20 contradicts "seniority levels blurred":** three of four boundaries *sharpened*. Only mid-senior ↔ director blurred, driven by `tech_count` sign flip (directors recast as tech leads).

**T28 produces the within-archetype credential-stack convergence claim.** The original junior-narrowing story fails every macro-robustness check; what survives is within-archetype credential-stack convergence in 10/10 archetypes.

**T23 inverts RQ3.** Worker any-use ~81%, posting broad AI 28.6%, hard requirement 6.0%. **The direction is the opposite of the original hypothesis.**

**T29 LLM authorship detection — the unifying mechanism.** 88.7% of 2026 postings score above the 2024 median on authorship style. AI rise 0-7% attenuation (real); length and requirement-breadth 23-62% attenuation (mostly style). **Length growth is recruiter-LLM drafting, not scope inflation.**

**Second V2 verification pass (Gate 3 verification):**

- 8/10 headlines PASS.
- **Three Gate 3 narrowings:**
  1. The T29 "length flips to −411 chars under style matching" is feature-set dependent. Cite attenuation, not sign flip.
  2. The T28 "credential-stack flips in 7/10 archetypes" is pattern dependent (V2 gets 2/10). Cite convergence in 10/10, flip count range 2-7.
  3. The "108× ChatGPT worker-to-posting ratio" has a near-zero denominator. Soften to "~10-15× for major AI tools."

See: [`audit/memos/gate_3.md`](../audit/memos/gate_3.md), [`gate_3_corrections.md`](../audit/memos/gate_3_corrections.md), [`V2_verification.md`](../audit/verifications/V2_verification.md)

## Gate 4 — Wave 4 synthesis

Final integration. Four core findings ranked. All Gate 2 V1 corrections and Gate 3 V2 narrowings applied throughout. SYNTHESIS.md written as the single handoff document for the analysis phase.

**Research questions at the end of the exploration:**

- **RQ3 (lead):** posting language LAGS worker AI adoption by roughly an order of magnitude. Direction inverted from the original RQ3.
- **RQ1 (reframed):** narrow within-archetype credential-stack convergence, not aggregate junior narrowing. Aggregate junior-share null survives every robustness check as a null.
- **RQ2 (reframed):** senior technical orchestration specialization, specifically at the mid-senior and director level, localized to the AI × senior interaction cell. Not a corpus-wide seniority blur.
- **RQ4 (unchanged):** qualitative interviews, five probes prepared (T25 artifacts).
- **RQ5 (new):** specification dependence as a methodological contribution.
- **RQ6 (new):** entry-specialist intermediaries as a structural component of the entry pool.

See: [SYNTHESIS.md](SYNTHESIS.md) for the full 500-line handoff.

## Three dominant framings overturned

Summary of what the exploration had to unlearn:

| Original framing | Verdict | What replaces it |
|---|---|---|
| "Junior rung is narrowing" | Failed every robustness check | Within-archetype credential-stack convergence (narrow) |
| "Seniority levels blurring" | 3/4 boundaries SHARPENED | Director-level tech-count sign flip, localized blur with mechanism |
| "Employers demand AI faster than workers adopt" | Direction INVERTED | Posting language is a lagging indicator; workers lead by ~10× |

Plus one mechanism overturned:

| Original mechanism | Verdict | What replaces it |
|---|---|---|
| "Length growth = scope inflation" | 23-62% attenuates under style matching | Recruiter-LLM drafting style migration |
