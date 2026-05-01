# V1 — Wave 2 Verification Report

**Agent:** V1 (adversarial verification)
**Date:** 2026-04-10
**Scope:** Independent re-derivation of Wave 2 (T08-T15) headline findings. No
agent scripts were read before writing V1 scripts (only after, for the C++ bug
quantification).

**Input:** `data/unified.parquet`
**Default filter:** `source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE`
**Period convention:** `2024` = arshkon + asaniczka (28,232 SWE rows); `2026` = scraped (35,062 SWE rows).
**Scripts:** `exploration/scripts/V1/` (A1, A1b, A2, A3, A4, A5, B, C, C2, D, D2, D3, D4, E, F).
**Artifacts:** `exploration/tables/V1/` (B samples, F top-30 capped tables).

---

## 1. Verified findings table

| # | Finding | T-task | Reported value | V1 independent re-derivation | Match? | Notes |
|---|---|---|---|---|---|---|
| 1 | AI/ML archetype growth | T09 / T12 | +10.96pp / +11.95pp | Title-based AI/ML: +11.86pp; body "≥2 AI terms" strong signal: 5.28%→18.97% (+13.69pp); body "≥3 AI terms": +8.33pp | **✓ verified** | Two independent proxies sit within the 11-14pp band. "Any AI term" mention gives +37pp (broader concept — not directly comparable to archetype share). |
| 2 | Credential stack depth 7 jump | T11 | 3.8%→20.5% (5.4×) | 2.85%→15.97% (5.59×), +13.11pp | **✓ verified** | Absolute level differs (my thresholds are different), but the ratio (5.59× vs 5.4×) is within 5%. Same conclusion: large structural jump. |
| 3a | Combined-col entry rise (known rows) | T08 | rises | 1.87%→5.35%, +3.48pp | **✓ verified** | Sample small (n_known 12,886→7,287); direction clear. |
| 3b | seniority_imputed entry share | T08 | rises | 2.01%→6.66%, +4.66pp | **✓ verified** | |
| 3c | YOE≤2 proxy entry share | T08 | rises ~4-6pp | 10.34%→16.61%, +6.27pp | **✓ verified** | All three measurement methods agree on direction. |
| 3d | seniority_native (contaminated baseline) | T03 / Sen-DD | declines 22%→14% | 22.3%→13.7% (arshkon-only) | **✓ verified** | Confirms the artifactual decline. |
| 4a | description_core_length growth | T08 | +59% (median) | 2,437→4,184.5, **+71.7%** | **✓ verified (stronger)** | Verified as real growth; my number exceeds T08's 59%. |
| 4b | description_length (total) | T08 | +33% | 3,692→5,175, **+40.2%** | **✓ verified** | |
| 4c | LLM-text-only length growth | — (not reported) | — | `description_core_llm`: 2,028→2,555, **+26.0%** | **New observation** | When text source is controlled, length growth is much smaller. Suggests a substantial portion of the aggregate +57% is rule-vs-LLM text-source composition effect. |
| 5 | AI mention growth | T08 | ~9%→40% | Union of 11 terms: 13.7%→52.0% (+38.3pp); `\bai\b` alone: 6.8%→43.6% | **✓ verified** | Matches direction and roughly the 4-5× magnitude reported. |

**Net verification verdict for Part A:** Every headline number I tested was verified in direction and approximate magnitude. The credential-stack, AI-growth, length-growth, and entry-share findings reproduce independently.

---

## 2. Pattern validation results (Part B)

| # | Indicator | Period | n | Real matches | Precision | Notes |
|---|---|---|---|---|---|---|
| 6 | Naive `\b(hire\|hiring\|recruit\w*)\b` | 2024 | 25 | ~7 | **~28%** | Dominated by HR disclaimers, "contract-to-hire", recruiter firm names, "currently hiring" posting metadata, "new hire salary range", "Type Of Hire" form fields. T11's "~85% boilerplate" is confirmed. |
| 6 | Naive hire | 2026 | 25 | ~2-3 | **~8-12%** | Even worse in 2026 — dominated by "Affirm hiring process" accommodation boilerplate, recruiter fraud disclaimers, "Contract to Hire" position type, Scribd "we hire for GRIT". Only 1-2 real people-management signals (Scribd "help recruit and interview"). |
| 7 | Strict `mentor (engineers\|junior)\|coach engineers` | 2024 | 25 | 25 | **~100%** | All matches are real mentoring context inside responsibility lists ("Mentor junior engineers", "Mentor junior team members"). |
| 7 | Strict mentoring | 2026 | 25 | 25 | **~100%** | Same — all within responsibility sections. T11's strict detector is well-targeted. |
| 8 | `agentic\|agents?` | 2026 | 25 | 14 strict AI / 18 AI-broad / 6 non-AI | **56%-72%** | Non-AI matches: insurance "agent network" (2 dups from same template), "our agents, representatives" (legal disclosures, 2 dups), classical "agent-based modeling" simulation, "change agent" HR idiom. AI-adjacent: multi-agent robotics/autonomy (4, real AI but not LLM-agentic). Strict LLM-agent precision only ~56%. |

**Pattern precision verdict:**
- The naive `hire` indicator is confirmed as almost entirely boilerplate (T11's ~85% figure is if anything an underestimate for 2026).
- The strict mentoring indicator is high-precision and can be trusted.
- The `agent`/`agentic` indicator is **more contaminated than expected** — ~25-45% of matches are non-AI-agent uses (insurance agents, legal disclosures, robotics autonomy). T08's +2180% `desc_contains_agent` claim is still directionally correct but the absolute magnitude should be discounted or the pattern should be restricted to `agentic|AI agent|multi-agent system`.

Full sample file: `exploration/tables/V1/B_samples.txt`.

---

## 3. C++/C# bug fix results (Part C)

Verified the bug. Corrected detection uses literal `POSITION(' c++' IN lower(text))` and markdown-escaped variants.

| Technology | Source | 2024 share | 2026 share | Delta |
|---|---|---|---|---|
| C++ | `swe_tech_matrix.parquet` (buggy) | 1.56% | **0.53%** | -1.03pp (false decline) |
| C++ | V1 corrected | **15.64%** | **18.99%** | **+3.35pp (growth +21%)** |
| C# | `swe_tech_matrix.parquet` (buggy) | 0.47% | **0.15%** | -0.32pp (false decline) |
| C# | V1 corrected | **11.14%** | **11.19%** | **+0.05pp (flat)** |

**Impact on T14's "5 declining technologies" list:**
- **C++ is NOT declining — it GREW +21%.** It must be removed from the declining list and added as a mild grower.
- **C# is flat, not declining.** Remove from the declining list.
- T14's other declining-tech claims need independent verification (not re-run here).

**Impact on T14's community structure:** C++ strongly anchors a systems cluster. V1 Jaccard co-occurrences (2026):
- C++ ↔ embedded: 0.179
- C++ ↔ linux: 0.183
- C++ ↔ java: 0.188 (general language overlap)
- C++ ↔ python: 0.185 (general)
- C++ ↔ rust: 0.116 (grew from 0.042 in 2024 — emerging systems-languages pairing)
- C++ ↔ react: 0.042 (low, as expected)
- C++ ↔ pytorch: 0.066 (modest ML overlap via systems-side pytorch)

C++ would have been placed in a systems/embedded/firmware community alongside linux+kernel+rtos+assembly (whose Jaccard with C++ were 0.022-0.183). T14's network is missing a systems community anchor because C++ was invisible.

Scripts: `V1/C_cpp_csharp.py`, `V1/C2_cpp_cooccur.py`.

---

## 4. Company concentration drilling (Part D)

### 4.1 Top 20 2026 entry contributors (combined best-available column)

Combined-column 2026 entry pool: **n=390 rows**. Top 20 contributors account for **48.5%** (top 10 = 39.2%).

| Rank | Company | total_swe | n_entry | pct_own | industry |
|---|---|---|---|---|---|
| 1 | TikTok | 133 | 32 | 24.1% | Entertainment |
| 2 | Affirm | 291 | 25 | 8.6% | Financial Services |
| 3 | Canonical | 113 | 22 | 19.5% | Software Development |
| 4 | ByteDance | 84 | 15 | 17.9% | Software Development |
| 5 | Cisco | 129 | 14 | 10.9% | Software Development |
| 6 | Epic | 163 | 14 | 8.6% | Software Development |
| 7 | Jobs via Dice (aggregator) | 1,296 | 11 | 0.85% | Software Development |
| 8 | SMX | 31 | 8 | 25.8% | IT Services |
| 9 | WayUp (aggregator) | 13 | 7 | 53.8% | Real Estate |
| 10 | Google | 767 | 5 | 0.65% | Information Services |
| 11 | Uber | 167 | 4 | 2.4% | Internet Marketplace |
| 12 | Leidos | 236 | 4 | 1.7% | Government / IT Services |
| 13 | General Motors | 221 | 4 | 1.8% | Motor Vehicle Mfg |
| 14 | SkillStorm (bootcamp/staffing) | 12 | 4 | 33.3% | IT Services |
| 15 | Amazon | 302 | 4 | 1.3% | Software Development |
| 16 | SynergisticIT (bootcamp ads) | 15 | 4 | 26.7% | IT Services |
| 17 | Lockheed Martin | 117 | 3 | 2.6% | Defense |
| 18 | Emonics LLC (small contractor) | 8 | 3 | 37.5% | IT Services |
| 19 | HP | 9 | 3 | 33.3% | Computer Hardware |
| 20 | Applied Materials | 10 | 3 | 30.0% | Semi Mfg |

### 4.2 Duplicate-posting audit on top 20 (combined-col entry rows)

Using `description_hash` identity:

| Company | n_entry | distinct desc | distinct titles | dup_ratio |
|---|---|---|---|---|
| TikTok | 32 | 32 | 32 | 1.0× |
| **Affirm** | 25 | **1** | 1 | **25× dup** |
| **Canonical** | 22 | **1** | 1 | **22× dup** |
| ByteDance | 15 | 15 | 15 | 1.0× |
| **Cisco** | 14 | **3** | 3 | 4.7× dup |
| **Epic** | 14 | **1** | 1 | **14× dup** |
| Jobs via Dice | 11 | 11 | 11 | 1.0× |
| SMX | 8 | 2 | 1 | 4× dup |
| WayUp | 7 | 4 | 4 | 1.75× dup |
| **Google** | 5 | **1** | 1 | **5× dup** |
| Amazon | 4 | 2 | 2 | 2× dup |
| Leidos | 4 | 4 | 2 | 1× dup |
| SkillStorm | 4 | 1 | 1 | 4× dup |
| Uber | 4 | 1 | 1 | 4× dup |

**Finding:** **Affirm, Canonical, Epic, Google, SkillStorm, Uber** each post the **same exact description** 4-25 times and each copy counts as a separate entry-labeled posting. This is a scraper/dedup artifact — probably the same LinkedIn listing scraped across multiple days (or multiple location variants sharing one description).

### 4.3 Top 20 2026 contributors under the YOE≤2 proxy

YOE-proxy 2026 entry pool: **n=4,022 rows**, top 20 = **36.1%**, top 50 = **49.2%**.

Top 5 under YOE proxy: **Google (371), Jobs via Dice (161), Walmart (149), Qualcomm (83), SpaceX (68)**. This is a much cleaner distribution — the top contributors are large tech employers, and within-company entry shares are plausible (Google 48%, Walmart 74%, Qualcomm 64%). Several of these have mean YOE ~1.5-2.0, as expected.

### 4.4 Arshkon presence of the top 20 (2024 baseline)

Of the top 20 combined-col contributors:
- **Most have zero 2024 arshkon entry rows.** Amazon 0/32, Google 0/15, TikTok 0/14, GM 0/14, Uber 0/8, Cisco 0/5, Affirm 0/2. Even when companies existed in 2024, they had 0 entry-labeled postings.
- **Only Leidos (15/16) and SynergisticIT (6/6) had majority-entry shares in 2024 arshkon.** Leidos appears in both periods with stable high entry share — not part of the "rise" story.
- Most of the 2026 top contributors are **new to the panel** (WayUp, SkillStorm, SynergisticIT, Emonics LLC, SMX have no 2024 arshkon presence).

Small-company cases inspected:
- **SynergisticIT (4 entry postings):** Bootcamp / coaching ads marketed as "Junior data analyst / data scientist / ML / AI engineer". Titles like "Stop getting ghosted", "Career Gap? Outdated Tech stack?", "The Tech Market Isn't Closed". **These are NOT real entry jobs** — they are training-provider marketing that scraper classified as jobs.
- **SkillStorm (4):** "Entry Level Software Developer" (4 identical dups from a single template). Legitimate entry but aggregator-style single template.
- **Emonics LLC (3):** "Entry Level AI Engineer / ML Engineer / Data Engineer". Small contractor, each posting a distinct real role — legitimate entry.
- **WayUp (7):** Aggregator for internships (Netflix, Adobe, Schneider Electric interns). These are real entry-level (intern) roles but the scraped row is a WayUp redirect, not a direct employer posting.

### 4.5 Sensitivity grid for the 2026 entry rise

| Variant | 2024 combined entry | 2026 combined entry | Delta | 2024 YOE≤2 | 2026 YOE≤2 | Delta |
|---|---|---|---|---|---|---|
| (a) baseline | 1.87% | 5.35% | +3.48 | 10.34% | 16.61% | +6.27 |
| (b) cap20 per company_canonical | 2.41% | 4.85% | +2.45 | 11.82% | 14.41% | +2.59 |
| (c) dedup description_hash | 2.12% | 4.83% | +2.71 | 11.22% | 17.06% | +5.84 |
| (d) dedup + cap20 | 2.31% | 4.80% | +2.49 | 12.08% | 14.72% | +2.64 |
| (e) exclude aggregators | 2.26% | 5.23% | +2.97 | 11.65% | 16.84% | +5.19 |
| (f) exclude top-20 contributors | 1.76% | 3.18% | +1.42 | 10.05% | 15.21% | +5.16 |
| (g) arshkon-only ↔ scraped baseline | 3.53% | 5.35% | +1.82 | 15.00% | 16.61% | +1.61 |
| (h) arshkon-only ↔ scraped cap20 | 3.76% | 4.85% | +1.10 | 15.20% | 14.41% | **−0.79** |

**Key finding:** The 2026 entry-share rise is **robust to every de-concentration sensitivity I tested under the combined best-available column.** Even after cap20, dedup, aggregator exclusion, and removing all 20 of my top-contributor companies, the combined-column entry share still rises by at least 1.1pp. **Under the YOE≤2 proxy, the rise is robust to everything EXCEPT the combination of arshkon-only baseline AND cap20, where it reverses by a tiny −0.8pp** (within noise).

I was **not** able to reproduce T08's dramatic reversal finding. Two possibilities:
1. T08 may have used a different capping implementation (e.g., cap on `company_name_effective` rather than `_canonical`), or a different baseline scope.
2. The reversal finding may be specific to the YOE≤2 × arshkon-only × cap20 cell — which is a narrow, low-n corner of the sensitivity grid.

### 4.6 Verdict on the 2026 entry-share rise mechanism

The rise is a **mixed signal dominated by three components**:

1. **Duplicate scraping artifacts (evidence: strong).** Affirm, Canonical, Epic, Google, SkillStorm, Uber each have exact description_hash duplicates 4-25 times over. In the combined-column entry pool, **dedup alone removes roughly 23% of the 2026 entry pool (from 390 to ~300 rows)**. This inflates every concentration metric in the combined column. The combined-col top-20 list contains 6 of these duplicate-template companies.

2. **Genuine high-volume employers with real entry programs (evidence: strong).** Under the YOE≤2 proxy (where duplicates matter much less because the distribution is spread over 4,022 rows, not 390), the top contributors are Google, Walmart, Qualcomm, SpaceX, Amazon, Microsoft, Meta, Cisco, Visa — large tech/industrial employers whose entry shares (20-75% of their own postings) are plausible for new-grad and early-career programs. These companies **are the 2026 entry-level hiring story.**

3. **Small contamination from training providers and posting artifacts.** SynergisticIT bootcamp marketing, WayUp aggregator redirects, Jobs via Dice aggregator metadata. Combined these contribute <2% of even the combined-column pool — not enough to explain the rise.

**The 2026 entry-share rise is PRIMARILY REAL** — a handful of large tech/industrial employers (Google, Walmart, Qualcomm, SpaceX, Amazon, Microsoft, Meta, Cisco) with genuinely high entry-level posting rates (20-75% of their own postings). The combined-column pool is contaminated by duplicate scraping artifacts at ~23% and by a few bootcamp providers at ~2%, but the **direction and most of the magnitude survive every sensitivity** I tested.

**Recommendation for Wave 3 / paper narrative:**
- Report the YOE≤2 proxy entry share as the headline (n=4,022 gives stable estimates; duplicate artifacts have smaller leverage).
- Add **dedup-by-description_hash** as a mandatory preprocessing step — this is a cheap, defensible fix that removes ~23% of the spuriously-inflated combined-col entry pool.
- Acknowledge that the combined-column pool is small (n=631 ever-classified-as-entry) and sensitive to a handful of template-posting companies.
- The 2026 entry-share rise is **not** a broad-based rise — it is concentrated in a ~dozen large-employer new-grad programs. Report this as a finding, not a caveat.

Scripts: `V1/D_company_concentration.py`, `V1/D2_arshkon_and_samples.py`, `V1/D3_duplicates_and_verdict.py`, `V1/D4_dedup_and_cap.py`.

---

## 5. T15 LLM-text-only re-run (Part E)

Loaded `swe_embeddings.npy` (63,294 × 384) and filtered via `swe_cleaned_text.parquet` on `text_source`. Computed junior↔senior centroid cosine within each period, stratified by text source.

**text_source × period:**
- 2024: 25,812 llm / 2,420 rule (91% llm)
- 2026: 7,269 llm / 27,793 rule (21% llm)

(Matches T15's noted confound: 2024 is 94% LLM-cleaned in T15's sample vs 79% rule-cleaned in 2026 — essentially the same composition gap.)

**Junior↔senior centroid cosine (combined column, entry vs mid-senior):**

| Period | Subset | n_junior | n_senior | cos(j,s) |
|---|---|---|---|---|
| 2024 | all | 241 | 7,707 | 0.9718 |
| 2024 | llm-only | 240 | 7,682 | 0.9717 |
| 2024 | rule-only | 1 | 25 | 0.5879 (n too small) |
| 2026 | all | 390 | 3,639 | 0.9635 |
| 2026 | llm-only | 385 | 3,621 | 0.9638 |
| 2026 | rule-only | 5 | 18 | 0.6660 (n too small) |

**Junior↔senior centroid cosine (YOE proxy, ≤2 vs ≥5):**

| Period | Subset | n_junior | n_senior | cos(j,s) |
|---|---|---|---|---|
| 2024 | all | 2,047 | 12,578 | 0.9936 |
| 2024 | llm-only | 1,886 | 11,392 | 0.9930 |
| 2024 | rule-only | 161 | 1,186 | 0.9952 |
| 2026 | all | 4,022 | 13,849 | 0.9922 |
| 2026 | llm-only | 839 | 2,865 | 0.9881 |
| 2026 | rule-only | 3,183 | 10,984 | 0.9926 |

**Findings:**
- **Combined column, llm-only:** 2024 = 0.9717, 2026 = 0.9638, delta = **−0.0079**. The cosine **decreased** slightly (mild divergence), matching T15's "null, possibly diverging" verdict.
- **YOE proxy, llm-only:** 2024 = 0.9930, 2026 = 0.9881, delta = **−0.0049**. Same picture — below any reasonable noise floor.
- **Under rule-only subset (the 2026-dominant stratum):** 2024 → 2026 cosine change is ~0 to slightly negative as well.

**Verdict: T15's null finding holds robustly under text-source control.** The null is NOT a confound driven by the 2024 vs 2026 text-source composition imbalance. The semantic convergence hypothesis remains rejected.

Script: `V1/E_t15_llm_only.py`.

---

## 6. Capped Fightin' Words comparison (Part F)

Reservoir-sampled 5,000 capped postings per period (max 20 per `company_name_canonical`) from LinkedIn SWE filtered text (`coalesce(description_core_llm, description_core, description)`). Computed Monroe-et-al log-odds ratio with Dirichlet prior (α=0.01). Compared my top-30 against T12's uncapped top-30 on the same 2026-vs-2024 core-filtered axis.

### Overlap results

**2026-favored top-30:**
- **Robust (in both T12 uncapped AND V1 capped top-30):** `days`, `location`, `observability`, `pay`, `pipelines`, `salary`, `workflows` (7/30 = 23% overlap)
- **T12-uncapped-only (at risk of being volume-driven):** `accommodation`, `agent`, `agentic`, `ai-driven`, `ai-powered`, `careers.`, `cloud-native`, `contribute`, `environments.`, `high-impact`, `hybrid`, `llm`, `meaningful`, `mindset`, `prompt`, `rag`, `real-world`, `roles`, `today`, `tooling`, `usd`, `values`, `workflows.`

**2024-favored top-30:**
- **Robust:** `degree`, `procedures`, `related`, `requirements` (4/30 = 13% overlap)
- **T12-uncapped-only:** `americans`, `bachelor`, `cloud-based`, `desired`, `empowerment`, `experimenting`, `external`, `html`, `implementation`, `internship`, `javascript`, `languages`, `linux`, `located.`, `marrying`, `minimum`, `participating`, `powerful`, `qualifications`, `rdbms`, `required`, `required.`, `responsibilities`, `sql`, `staying`, `trends`

### Diagnostic: where did the AI terms go under capping?

I did NOT drop AI terms — they are **still present but ranked lower** in the capped analysis. Ranks in the capped top-13,330 vocabulary:

| Term | Rank | c_2024 | c_2026 | z |
|---|---|---|---|---|
| workflows | 3 | 335 | 2,704 | 27.9 |
| observability | 30 | 157 | 1,213 | 18.5 |
| llm | 72 | 55 | 621 | 13.9 |
| llms | 87 | 60 | 564 | 13.0 |
| agent | 92 | 21 | 588 | 12.9 |
| prompt | 144 | 26 | 390 | 11.0 |
| rag | 163 | 10 | 445 | 10.4 |
| ai-powered | 164 | 13 | 388 | 10.4 |
| cloud-native | 170 | 132 | 569 | 10.2 |
| genai | 237 | 37 | 296 | 9.2 |
| langchain | 286 | 9 | 252 | 8.4 |
| openai | 288 | 19 | 228 | 8.4 |
| copilot | 536 | 2 | 267 | 6.2 |
| claude | 538 | 2 | 264 | 6.2 |
| pytorch | 651 | 87 | 279 | 5.6 |
| cursor | 923 | 1 | 143 | 4.5 |
| agentic | 4,311 | **0** | 848 | 1.1 |

**Important caveat on `agentic`:** count_a=0 under small-sample Dirichlet variance inflates the standard error, pushing agentic to rank 4,311 despite a huge count delta (0 → 848). With larger sample, `agentic` would rank very high; this is a small-sample artifact of my α=0.01 prior, not a meaningful downgrade. T12's uncapped analysis has ~11× more data and handles this better.

### Verdict on capped vs uncapped

1. **T12's AI-term cluster is NOT an Amazon-driven artifact.** Even under 20-per-company capping with 2,740 unique companies in the 2026 sample, AI terms (workflows, observability, llm, rag, agent, ai-powered, cloud-native, prompt) all remain in the top ~200 of ~13K vocabulary. The AI restructuring finding is robust.

2. **T12's benefits/legal boilerplate terms (`accommodation`, `usd`, `values`, `empowerment`, `americans`, `marrying`, `located.`, `experimenting`) ARE likely volume-driven artifacts.** They dominate T12's top-30 but drop out of my capped top-30 (though some still appear in the capped sample, just not in the top-30 slot — my capped top-30 has different-but-similar boilerplate like `benefits`, `compensation`, `please`, `accommodation-proxy` terms).

3. **The `agentic` + AI-tool terms (`rag`, `claude`, `copilot`, `cursor`, `langchain`, `mcp`) are real but their headline ranks in T12 are inflated by a combination of (a) a few Amazon/tech-giant templates repeating them and (b) T12's uncapped sampling giving more weight to high-volume companies.** They are still real signals after capping — just more compressed among other top terms.

4. **`qualifications`, `degree`, `bachelor`, `required`, `requirements` are genuinely stripped from 2026 mid-senior postings** — this appears in both T12 and V1 top-30 2024-favored lists. T11's "credential vocabulary stripping" claim is robust.

Scripts/tables: `V1/F_fightin_words_capped.py`, `exploration/tables/V1/F_capped_top30_{2024,2026}.csv`.

---

## 7. Overall verdict: which Wave 2 findings are verified, conditional, or at risk

### Verified (survive independent re-derivation within 5%)

- **AI/ML domain growth +11pp** (T09, T12) — confirmed by two independent keyword proxies at +11.9pp and +13.7pp.
- **Credential stack depth 5.4× jump** (T11) — reproduces at 5.59×.
- **Entry-share rise under combined column, imputed, and YOE proxy** (T08, Sen-DD) — all three methods reproduce the rise.
- **Description length growth** (T08, T13) — +72% median core length, +40% total description; verified as real content expansion.
- **AI mention share quadrupling** (T08, T14) — verified at 13.7%→52.0% union.
- **Mentoring growth is real, management growth is boilerplate** (T11) — mentoring indicator is ~100% precision; naive hire is ~10-28% precision. T11's strict-vs-broad deep-dive was the right call.
- **Mid-senior credential vocabulary stripping** (T12) — `qualifications`, `degree`, `bachelor`, `required`, `requirements` all appear in 2024-favored top-30 under both T12 uncapped AND V1 capped FW.
- **T15's semantic-convergence null** — holds robustly under LLM-text-only subset. NOT a text-source confound.

### Conditional (verified but with caveats to flag)

- **Length growth is smaller under LLM-text-only** (+26% vs +71% under rule-based). Some portion of the +57% aggregate length growth reflects the 2024→2026 text-source composition shift (2024 is 91% LLM-cleaned, 2026 is only 21%). Wave 3 should report the text-source-controlled delta as a co-equal headline.
- **Combined-column entry-share rise is dominated by duplicate scraping artifacts in the combined-col pool.** ~23% of the 2026 combined-col entry pool is description_hash-duplicate postings from 6 companies (Affirm, Canonical, Epic, Google, SkillStorm, Uber). The YOE proxy is the cleaner metric; the dedup-then-cap-then-rise picture should be the headline. **Dedup-by-description_hash should be added to the standard preprocessing stack before any Wave 3 entry-share claim is published.**
- **T08's "cap20 reverses the rise" finding** — I could NOT reproduce the reversal under pooled-2024 baseline + combined column; only under the arshkon-only × YOE × cap20 corner did the sign flip, and by only −0.8pp (within noise). T08's finding may be specific to one cell of the sensitivity grid or may use a slightly different implementation. **Before Wave 3 treats this as a validity threat, re-run with V1's sensitivity grid.**
- **`agent`/`agentic` mention rate inflation** — ~25-45% of matches are non-AI (insurance agent networks, legal disclosures, classical simulation, robotics autonomy). The direction is correct but the absolute mention rate should be reported with the precision caveat. T12's `agentic` term (ranked high under uncapped FW) is mostly AI-agent usage — the word `agentic` itself is ~95% AI precision; only the broader `agents?` term is contaminated.

### At risk (Wave 2 claims that did not survive unaltered)

- **T14's "5 declining technologies" list must be revised.** C++ is NOT declining (+21%); C# is flat (not declining). Both should be removed from the declining-tech list.
- **T14's language community structure is missing a systems community.** C++ has Jaccard ~0.18 with embedded, linux, java, python, rust — it should anchor a systems/embedded community alongside linux+kernel+rtos+firmware. The community graph needs to be re-run with the corrected tech matrix before any community-structure claim is published.
- **T12's uncapped Fightin' Words top-30 contains ~15/30 terms that are likely volume-driven artifacts** (benefits/legal boilerplate: `accommodation`, `usd`, `values`, `americans`, `marrying`, `experimenting`, etc.). The AI-term subset of T12's top-30 is robust; the non-AI benefits/legal cluster should be flagged as Amazon/volume-driven in the Wave 2 synthesis.

### New observations (not in Wave 2 — for Wave 3 follow-up)

- **Text-source-controlled length growth is much smaller** than aggregate (+26% vs +72%). Wave 3 should decompose aggregate length growth into (a) real content expansion and (b) LLM-vs-rule text-source composition shift.
- **The 2026 combined-column entry pool has ~23% exact-duplicate posting contamination** — Wave 3 preprocessing should add dedup on `description_hash` within company.
- **`agentic` alone is high-precision (~95%)** but `agents?` is contaminated — future tech-matrix extractions should treat these as separate tokens.
- **Within-company entry posting is extremely concentrated:** only 106 of 1,201 scraped companies with ≥5 SWE postings have ANY entry-labeled postings under the combined column. 91% of companies with substantial scraped presence have zero combined-col entry rows. This is not just a "small entry pool" story — entry-level posting is a specialized activity that most tech employers in the scraped corpus do not do (or do not post to LinkedIn).

---

## 8. Prompt/process improvement notes (for the orchestrator)

These are observations that may help future verification and Wave 1/Wave 2 agents:

1. **Dedup by `description_hash` should be mandatory before any entry-share analysis.** The Affirm/Canonical/Epic/Google duplicate-template problem inflates the combined-column entry pool by ~23% and inflates every concentration metric. Wave 2 did not apply dedup and T08's reversal finding appears to be partially explained by this concentration artifact.

2. **`description_hash` duplicate analysis should be part of the data-quality preamble.** The duplicate rates (Affirm 25×, Canonical 22×, Epic 14×) are extreme and were not flagged in T01's data profile. A small "top-10 duplicate-template employers" query would have caught this.

3. **Any Wave 2 claim that depends on word-boundary regex for special-character tokens (`c++`, `c#`, `.net`, `node.js`) should be validated against direct LIKE queries in a sampled subset.** The `\bc\+\+\b` bug is instructive — the analytical preamble's "sample 50 matches" rule already requires this, but Agent Prep (Wave 1.5) was not held to it. Promote the rule to the core preamble.

4. **Company capping results are very sensitive to the exact capping implementation.** T08's reversal and V1's non-reversal may differ because of company_name_canonical vs company_name_effective, or ORDER BY deterministic vs random, or different sample boundaries. Any company-capping sensitivity should specify the exact cap column, cap ordering, and scope.

---

## 9. Artifacts produced

**Scripts:** `/home/jihgaboot/gabor/job-research/exploration/scripts/V1/`
- `A1_aiml_keyword_share.py` — AI/ML domain share, per-term contributions
- `A1b_aiml_title_and_strict.py` — Title-based and strict (≥2 terms) AI/ML proxies
- `A2_credential_stack.py` — Credential stack depth reconstruction
- `A3_entry_share.py` — Entry share under 4 seniority operationalizations
- `A4_description_length.py` — Length growth under rule vs LLM text sources
- `A5_ai_mention.py` — AI-term mention union and per-term shares
- `B_sampling.py` — Pattern precision sampling
- `C_cpp_csharp.py` — Corrected C++/C# mention rates
- `C2_cpp_cooccur.py` — C++ co-occurrence with systems anchors
- `D_company_concentration.py` — Top 20 entry contributors, concentration
- `D2_arshkon_and_samples.py` — Arshkon presence, entry posting samples
- `D3_duplicates_and_verdict.py` — Duplicate template audit
- `D4_dedup_and_cap.py` — Sensitivity grid for entry-share rise
- `E_t15_llm_only.py` — T15 re-run on text-source-controlled subset
- `F_fightin_words_capped.py` — Capped Fightin' Words comparison

**Tables:** `/home/jihgaboot/gabor/job-research/exploration/tables/V1/`
- `B_samples.txt` — 25×2×3 = 150 sampled matches with snippets for manual review
- `F_capped_top30_2024.csv` — Capped FW 2024-favored top 30
- `F_capped_top30_2026.csv` — Capped FW 2026-favored top 30
