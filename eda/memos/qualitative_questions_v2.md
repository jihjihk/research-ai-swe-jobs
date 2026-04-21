# Qualitative interview questions for v2 EDA findings

Supplement to [`docs/2-interview-design-mechanisms.md`](../../docs/2-interview-design-mechanisms.md) (the canonical interview-protocol doc). This memo translates our v2 EDA verdicts (see [`../reports/open_ended_v2.md`](../reports/open_ended_v2.md)) into specific interview probes that can **verify** findings (adjudicate real vs artifact) or **explain** mechanisms (why did this happen?).

Organization is by **research theme**, not one-section-per-hypothesis, because related findings share probes. Each question is tagged with:

- **[cohort]** — A (senior SWEs / tech leads), B (juniors / CS students), C (hiring-side: EMs, recruiters, talent leads, founders). C is the critical adjudication cohort for JD authorship and employer intent.
- **[V]** verify — tests whether the finding is real vs a measurement artifact.
- **[E]** explain — tests mechanism / why.
- **[H*]** — which hypothesis verdict(s) this probe adjudicates. See priors memo for H1–H13 framing.

Questions slot into the existing 6 modules of the canonical protocol — themes 1 & 5 into Module 5 (JD authorship), theme 2 into Module 3 (junior bar), theme 3 into Module 2 (AI workflow), theme 4 into Modules 4–5, theme 6 into Modules 5 & 2.

---

## Theme 1 — Who rewrote the JDs, and why?

**What the data says (v2 headline H13):** On a 292-company panel of firms that posted SWE roles in *both* 2024 and 2026, mean within-firm AI-vocab rate rose +19.4 pp (core) / +20.7 pp (full unified). 75% of firms rose, 61% rose >10 pp. Microsoft +51, Wells Fargo +45, Amazon +36. Defense firms (Raytheon, Northrop) flat. This rules out between-firm composition churn as the dominant driver — the *same firms* rewrote their *own* postings. But the data cannot tell us **who inside the firm** did the rewriting or **why**.

**Interview goal:** identify the mechanism — was this leadership-driven? recruiter-tool drift? hiring-manager request? an LLM template?

1. **[C, V, H13]** Walk me through how a SWE JD got written at your company in 2026. Who typed the first draft? Who added AI-related bullets? Who signed off?

    *Follow-up probes:* was there a leadership directive to "mention AI" anywhere in 2025? Did the recruiting team roll out a new template? Did you use an LLM to help draft the post? Did legal or DEI review it?

2. **[C, E, H13]** When was the last time you compared a 2024 SWE JD at your firm to the same role in 2026? [Then show the paired-JD stimulus.] Which differences are *substantive* (reflecting different work) vs *stylistic* (reflecting a template refresh or LLM draft)?

    *Follow-up probes:* point to one new line and say "was this added because the work changed, or because the template changed?"

3. **[A, V, H13]** Have you personally rewritten a SWE JD in the last 12 months? If yes: what did you add, what did you remove, and why? If no: did someone else ask you to review a rewrite?

4. **[C, E, H6 BT density, H13]** [Show chart of Big Tech vs rest AI rate: 44% vs 27%.] Big Tech firms mention AI in their SWE postings at almost 2× the rate of other employers. Does this match what you see inside BT firms — real tool adoption? Or is it a PR / recruiting-brand phenomenon?

    *Follow-up probe for non-BT respondents:* if your firm doesn't mention AI in postings, is that because you *can't* (no tools deployed) or *won't* (deliberate differentiation)?

**Stimulus materials:**

- **Paired 2024↔2026 JDs from our own 292-firm panel.** *To build.* Pull 4 companies from `eda/tables/S17_within_firm_panel.csv` across a range: one frontier-AI lab (Anthropic), one finance (Wells Fargo / Capital One), one Big Tech (Microsoft / Amazon), one flat firm (Raytheon). Format per `exploration-archive/v8_claude_opus47/artifacts/T25_interview/paired_jds_over_time.md`.
- Reuses existing v8 T25 paired JDs (Ramsey Solutions, Teradyne, VDart, Ridgeline) — reference, don't duplicate.

---

## Theme 2 — Junior bar: did it really fall?

**What the data says (v2 headlines H8, H5a):** Classic scope inflation predicts rising junior YOE demands. Our LLM-YOE shows the opposite — junior mean fell 2.01 → 1.23 years, median 2 → 1. Senior median also dropped 6 → 5. AI-vocab adoption is uniform across seniority (junior 27%, mid 30%, senior 31% in 2026-04), falsifying the junior-first-automation story (H5a). But this is a descriptive statistic on extracted-YOE; real hiring bars could be moving differently.

**Interview goal:** distinguish real hiring-bar change from measurement drift in the "junior" label or LLM extractor. Test competing explanations: (i) AI-assisted juniors onboard faster so employers can afford to hire less-experienced candidates, (ii) cost pressure pushed firms to accept new-grads and apprentices, (iii) the LLM YOE extractor has a systematic bias, (iv) "junior" postings aren't actually being filled by the people we think.

5. **[C, V, H8]** Look at your team's entry-level hires from 2026 vs 2024. In years of prior professional experience — not counting internships — are your 2026 hires less experienced, more experienced, or about the same?

    *Follow-up probes:* if less experienced, was that a deliberate bar-lowering or were you taking what you could get? Did AI tooling make you more willing to train from zero?

6. **[C, E, H8, H_B contraction]** Our data shows 2026 junior postings ask for fewer years of experience than 2024 — median dropped from 2 to 1. Does that match your internal screening rubric? Or are you screening harder in other ways (school, coding test, AI-tool fluency) while dropping YOE explicitly?

    *Counter-probe:* are some 2026 "junior" postings actually filled by internal transfers, apprenticeship programs, or rotations — i.e., experienced people labeled junior?

7. **[B, V, H8]** When you apply to a junior SWE role in 2026, what does the posting *actually* ask for in terms of experience? Have you seen a 1-YOE-minimum posting go to someone with 3+ years?

8. **[A, E, H5a, H8]** If AI is automating routine junior work, you'd expect juniors to need *more* experience to get hired (they need to operate at a higher level from day one). Our data shows the opposite — junior YOE fell, and AI vocabulary is equally present at senior and junior levels. What's your interpretation?

**Stimulus materials:**

- **Junior YOE trajectory chart** — `eda/figures/S12_yoe_trajectory.png` already exists; use as-is.
- **Inflated junior JDs** — reuse v8 T25 `inflated_junior_jds.md` (4 entry-labeled postings with high scope). Reference, don't rebuild.

---

## Theme 3 — Which AI tools are actually used, and does "AI mention" signal a ghost job?

**What the data says (v2 headlines H9, H10):**
- **H9 vendor leaderboard (2026-04):** Copilot 4.25% > Claude 3.83% > OpenAI 3.63% > Cursor 2.17% > Anthropic 1.48%. Growth rates: Claude 190×, Cursor >200×, Copilot 53×, ChatGPT brand plateauing.
- **H10 AI ≠ ghost:** AI-mentioning SWE postings have *lower* inflated rate (4.5%) than non-AI (5.6%). Falsifies the intuition that AI buzzwords signal a fake posting.

Both are labor-demand-side inferences. They need cross-checking against what engineers actually experience.

9. **[A, V, H9]** Which AI coding tools do you use in a typical workday? How often? For what kinds of tasks?

    *Follow-up probes:* if you mention Copilot — when did your firm deploy it? if you mention Claude / Cursor — is it your personal choice or a firm-provided license? how does Claude in Cursor differ from Copilot for you?

10. **[A, E, H9]** We see Copilot mentioned in 4.25% of 2026 SWE postings, Claude in 3.83%, Cursor in 2.17%. Does that ranking match your on-the-ground experience of which tools have actually won in your workplace? Or are postings lagging real adoption?

11. **[C, V, H9, H10]** When a 2026 SWE posting says "experience with Copilot" or "familiarity with LLMs," do you actually screen for that? Can you give one concrete example of a candidate you rejected for lacking AI-tool fluency?

    *Red flag for interviewer:* if respondent says "yes we screen" but can't name a single concrete rejection, code as "aspirational / ornamental requirement." Ties to H10.

12. **[B, V, H10]** Have you ever applied to an AI-heavy SWE posting and suspected it was a ghost / not serious? What signals made you think that?

    *Counter-probe:* have you had an unusually responsive / fast interview process from an AI-heavy posting? (Ties to the H10 finding that AI postings may indicate *real* hiring intent rather than ghosting.)

**Stimulus materials:**

- **Vendor leaderboard chart** — `eda/figures/S13_vendor_mentions.png` already exists; use as-is.
- **v8 T25 `employer_usage_divergence.png`** — reuse for RQ3 framing (employers under-specify AI vs worker usage).

---

## Theme 4 — Where AI appears: Big Tech density and the finance/nuclear control-niche

**What the data says (v2 headlines H6, H11, H7):**
- **H6 BT AI density:** 44% at Big Tech (Google, Meta, Amazon, Apple, Microsoft, Oracle, Anthropic, OpenAI, etc.) vs 27% at the rest — a robust 17 pp gap. BT *posting share* also rose from 2.4% → 7.0%, opposite of the public layoff narrative.
- **H11 control-niche:** Within the control cohort (non-SWE occupations), the AI-rate rise concentrates in finance/accounting and electrical/nuclear engineering. Nursing, HR, sales, marketing, and retail are nearly untouched.
- **H7 SWE-specificity:** 23:1 delta ratio of SWE AI-rate rise over control AI-rate rise. Replicates v8 T18 on our cleaner sample.

**Interview goal:** disambiguate real tool adoption from branding / vendor-push phenomena, and understand why AI-language has crossed into *some* non-SWE occupations but not others.

13. **[A at BT, V, H6]** You work at a Big Tech firm. In the last 18 months, have SWE interview loops changed to explicitly test AI-tool fluency? Or is the posting language ahead of the actual screening?

14. **[C at BT, E, H6]** Our data shows BT firms mention AI in SWE postings at 1.6× the rate of the rest of the market. Three candidate explanations — tell me which ring true: (a) BT firms adopted AI tooling earliest so their SWE work genuinely changed more, (b) BT firms are under more pressure to signal modernity to investors / candidates, (c) BT firms have bigger recruiting teams with more template drift.

15. **[C outside tech, V/E, H11]** [For finance / nuclear engineering respondents.] Our data shows your industry is one of only two non-SWE sectors where AI language rose noticeably in 2026 (finance at X%, nuclear/electrical at Y%). Is that real — are you actually using AI tools for financial modeling / reactor monitoring / etc? Or are vendors pitching into your industry aggressively?

    *Follow-up probes:* did compliance reviewers push back on AI-in-JDs in your industry? (H11 hypothesis: regulated-work sectors adopt cautiously but earlier-than-retail once vendors show up.)

16. **[C, V, H7]** Does "AI changes SWE work" feel qualitatively different from "AI changes knowledge work generally" when you think about hiring? What occupations do you think will move next?

    *Counter-probe:* we see nursing and retail at near-zero AI mention rates in postings. Is that because AI genuinely hasn't hit those jobs, or because those industries aren't fashionable to mention it even where it exists?

**Stimulus materials:**

- **Control-family AI rates chart** — `eda/figures/S15_control_ai_drivers.png` already exists.
- **Big Tech vs rest chart** — `eda/figures/S10_core_bigtech_vs_rest.png` already exists.
- **SWE-vs-control divergence** — `eda/figures/S11_core_swe_vs_control.png`; reuses v8 T25 `ai_gradient_chart.png`.

---

## Theme 5 — The layoff-narrative layer of H1 (unresolved by quantitative EDA)

**What the data says (v2 headline H1):** We falsified H1 **at the content level** — if AI were narrative cover for unrelated macro layoffs, SWE and control postings would co-move. They don't (23:1 delta ratio). But we **did not test** the narrative layer: when firms publicly explained 2023–2025 cuts, did they invoke AI as the honest driver or as cover for cost/rate/outsourcing reasons? That question can only be answered qualitatively.

**Interview goal:** adjudicate whether AI-as-layoff-reason was honest or constructed, especially at firms The Economist named (Oracle, Block, Amazon, Meta).

17. **[C, V/E, H1-narrative]** When your firm announced layoffs or hiring freezes in 2023–2025 (or if not your firm — your peers' firms), what was the *internal* explanation vs the *external* one? Did "AI" feature in the internal rationale, the press release, or both?

    *Follow-up probe:* ask specifically about Oracle, Block, Amazon, Meta reductions — were people on those teams told AI was the reason?

18. **[A who survived a layoff cycle, E, H1-narrative]** What did your manager or skip-level say when your team was resized? What did laid-off colleagues say in exit conversations? Did AI feature prominently, weakly, or not at all?

19. **[C, E, H1-narrative, H9]** Put this together for me: the firms that announced AI-driven cuts — from what you saw, did they *already* have high AI tool adoption (= AI was real), or was the AI adoption announced simultaneously with the cuts (= AI was narrative)?

20. **[A/C, V, H1-narrative]** We tested whether AI was a narrative cover by looking at whether control occupations (nurses, accountants) also showed AI language rise. They didn't. Does that convince you the content-level story is real? What would change your mind?

**Stimulus materials:**

- **Headcount / posting-volume timeline by firm** — *to build (optional).* Requires pairing named-firm layoff dates with posting volume from `data/unified_observations.parquet` — a follow-up analysis, not in current v2.
- Use v8 T25 `employer_usage_divergence.png` and our S11 chart to anchor.

---

## Theme 6 — Posting-survival signals and the ghost-job market

**What the data says (v2 H12):** In 2026-03 (the one period with full observation cadence), AI-mentioning SWE postings persisted 0.9 days longer on average than non-AI postings (5.27 vs 4.36 days mean). Control postings with AI mention showed a larger gap (7.5 vs 2.6 days) but with small n. The direction is ambiguous — AI postings might persist longer because they're *harder to fill* (high demand, scarce talent) or because they're *not real hiring intent* (pipeline / compliance).

**Interview goal:** disambiguate the direction of the small survival effect. Combines with H10 (AI postings aren't ghost-like in the LLM sense) — which force dominates?

21. **[C, E, H12]** When an "AI engineer" or "AI-experience-required" SWE posting sits open for 10–20 days at your firm, what's typically the reason? Hard-to-fill, open pipeline, compliance, or something else?

22. **[C, V, H12, H10]** [Show composite paired chart of S14 (AI ≠ ghost) + S16 (AI persists longer).] Our data suggests AI-mentioning postings are *both* less ghost-like AND stay live longer. Does that combination make sense to you? What mechanism could produce it?

    *Candidate interpretations to probe:* (a) AI-required roles are genuinely in demand, so worth keeping open; (b) AI-required roles attract more applicants so are reviewed more carefully; (c) firms batch-interview AI-role candidates to compare; (d) AI postings are actually internal-promotion placeholders.

**Stimulus materials:**

- **Posting-survival chart** — `eda/figures/S16_posting_survival.png` already exists.
- **AI × ghost crosstab** — `eda/figures/S14_ai_ghost_crosstab.png` already exists.

---

## Cross-cutting infrastructure

### A. Mapping to the existing 3-cohort × 6-module protocol

| Theme | Primary module in `docs/2-interview-design-mechanisms.md` | Secondary module |
|---|---|---|
| 1 · Who rewrote the JDs | Module 5 (JD authorship) | Module 1 (shift as experienced) |
| 2 · Junior bar | Module 3 (junior bar) | Module 5 |
| 3 · Vendor tools | Module 2 (AI workflow) | Module 5 |
| 4 · BT / control-niche | Module 5 | Module 4 (senior role) |
| 5 · Layoff narrative | Module 5 | Module 1 |
| 6 · Posting survival | Module 5 | — |

Six themes × ~4 questions = ~24 new probes. The canonical protocol targets `n=24` total interviews (8 per cohort, 60 minutes each). Keep new probes below 10 min per cohort per interview so existing Module coverage isn't crowded out.

### B. Coverage by cohort

| Cohort | Themes where this cohort is primary | Themes where this cohort is secondary |
|---|---|---|
| **A — seniors** | 2 (junior bar), 3 (workflow), 5 (narrative) | 1, 4, 6 |
| **B — juniors** | 2 (junior bar), 3 (workflow) | — |
| **C — hiring-side** | 1 (JD authorship), 2 (junior bar), 4 (BT/niche), 5 (narrative), 6 (survival) | 3 |

Cohort C carries the heaviest load — consistent with the canonical doc's observation that the hiring-side cohort is essential for mechanism evidence.

### C. Stimulus material inventory

| Theme | Stimulus needed | Status | Source |
|---|---|---|---|
| 1 | Paired 2024↔2026 JDs from v2 panel (4 companies) | **To build** | `eda/tables/S17_within_firm_panel.csv` |
| 1 | v8 T25 paired JDs (Ramsey, Teradyne, VDart, Ridgeline) | Exists | `exploration-archive/v8_claude_opus47/artifacts/T25_interview/paired_jds_over_time.md` |
| 2 | YOE trajectory chart | Exists | `eda/figures/S12_yoe_trajectory.png` |
| 2 | Inflated junior JDs (4 postings) | Exists | v8 T25 `inflated_junior_jds.md` |
| 3 | Vendor leaderboard chart | Exists | `eda/figures/S13_vendor_mentions.png` |
| 3 | Employer-usage divergence | Exists | v8 T25 `employer_usage_divergence.png` |
| 4 | Control-family AI rates | Exists | `eda/figures/S15_control_ai_drivers.png` |
| 4 | Big Tech vs rest | Exists | `eda/figures/S10_core_bigtech_vs_rest.png` |
| 4 | SWE vs control (AI gradient) | Exists | `eda/figures/S11_core_swe_vs_control.png` + v8 T25 equivalent |
| 5 | Firm-specific posting volume + layoff timeline | **To build (optional)** | requires new analysis on `data/unified_observations.parquet` |
| 6 | Posting survival | Exists | `eda/figures/S16_posting_survival.png` |
| 6 | AI × ghost crosstab | Exists | `eda/figures/S14_ai_ghost_crosstab.png` |

Most themes are fully provisioned with existing figures. The two to-build items (theme 1 paired JDs from our panel; theme 5 firm-specific timeline) can be follow-up tasks.

### D. "What good answers look like" — guardrails against confabulation

General principles:

- **Specific beats general.** "Yes we use Copilot" is not enough; press for a specific code-review conducted that week. If they can't, code as aspirational.
- **Internal evidence beats external narrative.** "The press release said AI" is not evidence; the internal Slack explanation is. Probe for both.
- **Disagreement with the data is a feature, not a problem.** If a respondent says "no, we don't mention AI more than we used to" and our data shows their firm rose +30pp, that disagreement is itself a data point. Don't challenge; explore how they arrived at that belief (maybe they're not in the posting-review loop).
- **Never re-introduce a falsified hypothesis as the researcher's position.** If a respondent says "junior scope inflation is obvious," explore *how* they see it — don't correct them to "actually our data shows the opposite." Capture the belief; triangulate later.

Theme-specific red flags:

| Theme | Red flag for confabulation |
|---|---|
| 1 | Respondent takes credit for a rewrite they can't describe in concrete detail (e.g., "who did you add to the review loop?") |
| 2 | Respondent says "the bar rose" without being able to cite a specific candidate they would have hired 2 years ago and wouldn't hire now |
| 3 | Respondent names a vendor they "screen for" but can't name a candidate rejected for lacking that vendor's skills |
| 4 | Respondent attributes BT AI density to "because they can afford it" without any concrete adoption story |
| 5 | Respondent gives the external press-release version rather than an internal version |
| 6 | Respondent claims a posting was "kept open on purpose" but can't say who made that decision |

### E. Disclosure to respondents (ethics / framing)

The canonical protocol (Module 6 + Analysis Plan) addresses this, but v2 stimulus materials warrant an explicit note:

- Stimulus JDs are from public datasets (Kaggle 2024 snapshots + our LinkedIn scrape). We are not showing respondents their own firm's internal JDs unless they volunteer to.
- Charts are from our own analysis of ~110k LinkedIn postings. The methodology is documented in [`../reports/open_ended_v2.md`](../reports/open_ended_v2.md) and can be shared on request.
- Respondents should be told: *"We are interested in your interpretation of what the data shows, including where you disagree with it. There are no right answers."*

### F. Integration with analysis plan

These questions generate codes that map to the existing deductive code families in `docs/2-interview-design-mechanisms.md` §Analysis Plan:

- Theme 1 → `jd_authorship`, `anticipatory_restructuring`
- Theme 2 → `junior_scope_inflation`, `screened_vs_unscreened_requirement`, `career_ladder_breakdown`
- Theme 3 → `actual_ai_workflow_change`, `ghost_requirement`
- Theme 4 → `senior_archetype_shift`, `actual_ai_workflow_change`, `market_tightness_non_ai`
- Theme 5 → `anticipatory_restructuring`, `market_tightness_non_ai`, `zeitgeist_anxiety`
- Theme 6 → `ghost_requirement`, `jd_authorship`

No new code families required.

---

## Summary

24 distinct interview probes across 6 themes, all tagged with cohort, verify/explain, and finding(s). Uses existing v8 T25 stimulus materials where already provisioned; names 2 items to build (paired JDs from our 292-firm panel, firm-specific layoff timeline). Slots into the canonical 3-cohort × 6-module protocol without requiring protocol changes.

**Next actions** (if the qualitative phase moves forward):

1. Build the 4-company paired-JD pack from `eda/tables/S17_within_firm_panel.csv` (theme 1 stimulus).
2. Decide whether to build the firm-specific posting-volume + layoff timeline (theme 5 optional stimulus).
3. Merge relevant probes from this memo into `docs/2-interview-design-mechanisms.md` §"Suggested Question Bank by Cohort" — this is a separate edit on a different branch per current scope discipline.
4. Run 2 pilot interviews (per canonical protocol Week 1) to refine wording before formal data collection.
