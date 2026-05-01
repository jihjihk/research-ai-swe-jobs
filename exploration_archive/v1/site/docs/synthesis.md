# Exploration Synthesis

**Date:** 2026-03-23
**Coverage:** T01-T17 (16 data audit, validation, and analysis tasks + 1 artifact task)
**Purpose:** Single handoff document for the analysis agent. Read this first.

---

## 1. Data Quality Verdict per RQ

### RQ1: Employer-Side Restructuring (Junior Share, Scope Inflation, Senior Redefinition)

**Can we answer it?** Yes, with caveats. The headline finding is robust: SWE entry-level share declined from 21.0% to 13.9% between April 2024 and March 2026 (seniority_patched). This finding survives:
- Within-company decomposition: -9.3pp within 379 overlapping companies (T13)
- Cross-occupation test: Control group shows opposite trend (+21pp), ruling out macro confounding (T16)
- Company-capping sensitivity: <1.5pp shift after capping at 10 postings/company (T06)
- 3 of 5 seniority variants agree on decline direction (T09)

**Constraints:**
- Direction depends on seniority operationalization: 3 native-label variants show decline, 2 title-only variants show increase (T09). The native-label cluster has 5-8x more entry observations and is the recommended primary.
- Historical entry-level baseline rests entirely on arshkon (n=745 entry after patching). Asaniczka contributes zero entry-level labels (T03).
- seniority_llm has not run; the patched workaround is the best available but is not the definitive classification.
- Scope inflation is field-wide, not junior-specific (T12). Organizational language (cross-functional +10pp, ownership +8pp) increased at all seniority levels. But formal credential requirements (YOE, MS/PhD) did NOT inflate -- entry-level YOE actually decreased from 3.0 to 1.0 years (T11, T12).

### RQ2: Task and Requirement Migration

**Can we answer it?** Yes. Text-based analyses are well-powered and the key findings are strong:
- AI/ML terminology is the dominant temporal distinguisher across all comparisons (T10 Fightin' Words)
- AI prevalence surged 19% -> 55% (broad) and 3.6% -> 24% (narrow LLM-specific) (T11)
- Ownership/autonomy language shifted toward 2026 while traditional "management" language favors 2024 (T10)
- 74 newly emerging terms are dominated by AI tooling: agentic, claude, gemini, embeddings, mcp, cursor (T11)
- Entry-level JSD (0.059) is 52% higher than overall JSD (0.038), confirming entry-level postings changed disproportionately (T11)

**Constraints:**
- Description length grew 64% (median 3124 -> 5110). All keyword counts must be normalized by length; binary indicators are less affected but not immune (T05, T08).
- Boilerplate removal accuracy is ~44%. Use `description` (not `description_core`) for keyword/pattern work (T01).
- Skills fields (skills_raw) are not available cross-period. All requirement analysis must use description-text NLP (T03).

### RQ3: Employer-Requirement / Worker-Usage Divergence

**Can we answer it?** Yes. T14 provides clean results:
- Employer AI requirements tripled: 18% -> 55% (any AI), 3% -> 26% (AI tools) (T14)
- Worker-side benchmark (StackOverflow): 62% active use in 2024, 51% daily use in 2025
- Gap closed from ~40pp to ~0pp between 2024 and 2026 (T14)
- AI-tool-specific category (LLM/GPT/Copilot) is the cleanest variable: 3% -> 26% (T14)

**Constraints:**
- Worker-side data is survey-based (self-reported) and measures different constructs than posting keywords (T14).
- Cannot distinguish whether convergence is driven by same companies updating requirements vs. AI-heavy companies posting more in 2026 (compositional effect). T13's within-company analysis shows AI surge is +33pp within-company vs. +1.5pp composition, supporting genuine updating.

### RQ4: Mechanisms (Interview-Based)

**Can we answer it?** The quantitative foundation is strong. Five interview elicitation artifacts are ready (T17):
1. Scope-inflated entry-level JDs
2. Same-company paired JDs (2024 vs 2026)
3. Junior-share decline + AI model release timeline
4. Senior archetype shift chart
5. Employer-worker AI divergence chart

---

## 2. Recommended Analytical Samples

### Primary SWE sample

```sql
WITH swe_patched AS (
  SELECT *,
    -- seniority_patched CTE from exploration/seniority_workaround.sql
    ...
  FROM 'preprocessing/intermediate/stage8_final.parquet'
  WHERE source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
)
SELECT * FROM swe_patched WHERE is_swe = true
```

**Rows:** ~30,283 SWE (4,438 arshkon + 21,626 asaniczka + 4,219 scraped)

### Entry-level comparison sample (RQ1)

Use arshkon (2024-04) vs scraped (2026-03) only. Asaniczka cannot contribute to entry-level analysis.

| Source | Total SWE | Entry (patched) | Entry % | Known seniority |
|--------|-----------|-----------------|---------|-----------------|
| arshkon | 4,438 | 745 | 21.0% | 3,553 (80.1%) |
| scraped | 4,219 | 549 | 13.9% | 3,948 (93.6%) |

### Within-company panel (RQ1 identification strategy)

- 379 companies appearing in both arshkon and scraped
- 71 companies with 3+ postings in both periods (suitable for company fixed-effects)
- Within-company entry decline: -9.3pp (stronger than overall -3.8pp)

### Cross-occupation sample (T16)

| Group | arshkon n | scraped n | Purpose |
|-------|-----------|-----------|---------|
| SWE | 4,438 | 4,219 | Primary |
| SWE-adjacent | 1,397 | 598 | Technical gradient |
| Control | 8,849 | 4,086 | Macro confound test |

### Text analysis sample

- Use `description` (not `description_core`) for all keyword/pattern work
- Normalize keyword counts by `description_length`
- For binary indicators (any mention), length normalization is less critical but should be reported as sensitivity

### Key columns for analysis

| Need | Column | Notes |
|------|--------|-------|
| Seniority | `seniority_patched` | Via workaround CTE. Primary variable. |
| Seniority (coarse) | `seniority_3level_patched` | junior/mid/senior/unknown |
| Seniority (sensitivity) | `seniority_final` | Report as sensitivity variant |
| SWE classification | `is_swe` | 30,283 rows. Regex tier is cleanest. |
| Company | `company_name_canonical` | For grouping; `company_name_effective` for display |
| Aggregator | `is_aggregator` | Exclude in sensitivity |
| Period | `period` | 2024-01, 2024-04, 2026-03 |
| Description | `description` | Full text, not description_core |
| YOE | `yoe_extracted` | 16-40% coverage; sparse but directionally useful |

---

## 3. Seniority Column Recommendation

**Use `seniority_patched`** from `exploration/seniority_workaround.sql` as the primary analysis variable.

### Why

The pipeline's `seniority_final` has a native_backfill bug (T02) that causes it to ignore 9,490 SWE rows with usable `seniority_native` labels. The `seniority_patched` workaround applies the correct priority order:

1. Title-strong signals (title_keyword, title_manager) -- highest priority
2. Native backfill (seniority_native) -- second priority
3. Weak title signals -- third priority
4. Unknown -- fallback

### Impact

| Source | seniority_final unknown % | seniority_patched unknown % | Entry n (final) | Entry n (patched) |
|--------|--------------------------|----------------------------|-----------------|-------------------|
| arshkon | 54.6% | 19.9% | 89 | 745 |
| asaniczka | 29.3% | 0.0% | 124 | 121 |
| scraped | 44.6% | 6.4% | 132 | 549 |

### Sensitivity requirement

Always report results using both `seniority_patched` (primary) and `seniority_final` (sensitivity). The variant table from T09 shows:

| Variant | Entry direction | Entry n (arshkon/scraped) |
|---------|----------------|--------------------------|
| A: Patched (primary) | Declining -7.1pp | 745 / 549 |
| B: seniority_final (broken) | Increasing +1.2pp | 89 / 132 |
| C: Native only | Declining -7.4pp | 710 / 545 |
| D: High-confidence | Declining -7.3pp | 745 / 548 |
| F: Title-strong only | Increasing +1.6pp | 87 / 130 |

### Pending fix

Stage 5 `infer_seniority_family()` needs to align with `is_swe` classification (T02). After the fix, re-run preprocessing and use `seniority_final` directly. Also run `seniority_llm` (Stage 10/11) for an independent third signal.

---

## 4. Known Confounders

### 4.1 Description length growth

- Median description length grew 64% (3,124 -> 5,110 chars) from 2024 to 2026 (T05, T08)
- **Mitigation:** Normalize keyword counts by `description_length`. Use binary indicators (any mention) as primary; rate-per-1K-chars as sensitivity. T12 showed length-normalized tech density barely changed (1.2 -> 1.3 per 1K chars) while raw counts increased.

### 4.2 Asaniczka label gap

- asaniczka has zero entry-level native labels (T03). Its seniority distribution is 100% mid-senior + associate.
- **Mitigation:** Exclude asaniczka from all entry-level trend analyses. Use only arshkon (2024-04) vs scraped (2026-03) for seniority-stratified comparisons. Asaniczka is useful only for mid-senior subgroup analyses and as volume context.

### 4.3 Aggregator contamination

- Aggregator rates differ by source: asaniczka 28%, arshkon 12%, scraped 14.5% (T08)
- "Jobs via Dice" (130 scraped rows) likely an undetected aggregator (T06, T13)
- TEKsystems, Apex Systems show erratic seniority labels in overlap analysis (T13)
- **Mitigation:** Run all key analyses with `is_aggregator = false` as sensitivity check. Flag "Jobs via Dice" and "Turing" for exclusion.

### 4.4 Field-wide scope inflation

- Organizational language (cross-functional, ownership, end-to-end, collaboration) increased at ALL seniority levels, not just entry (T12)
- **Mitigation:** Report the field-wide baseline shift alongside entry-level-specific findings. The claim is not "employers are piling requirements onto juniors uniquely" but rather "the entire SWE profession is experiencing scope inflation, and entry-level roles are not exempt despite their lower formal requirements (YOE declined)."

### 4.5 Company concentration (asaniczka)

- Capital One + Recruiting from Scratch = 15.3% of asaniczka SWE postings (T06)
- **Mitigation:** Company-capped sensitivity (<1.5pp shift after capping at 10/company). Also report asaniczka results with and without top-2 companies.

### 4.6 SWE classification noise

- Overall FP rate ~5.5%; embedding_llm tier has 26% FP rate (T04)
- **Mitigation:** Regex-tier-only sensitivity (84% of SWE rows, ~2% FP). Report main results on full sample, sensitivity on regex-only.

---

## 5. Preliminary Findings for RQ1-RQ3

### RQ1: Employer-Side Restructuring

| Finding | Direction | Magnitude | Confidence | Source |
|---------|-----------|-----------|------------|--------|
| Entry-level share decline | Down | -7.1pp (21% -> 14%) | **Moderate-High** (3/5 variants agree; within-company -9.3pp; cross-occupation confirms) | T08, T09, T13, T16 |
| Scope inflation (organizational language) | Up | +8-24pp at entry level | **High** (robust across metrics) | T12 |
| Scope inflation (formal credentials) | Down/flat | YOE -2.0yr, MS/PhD flat | **Moderate** (YOE extraction sparse in 2026: 16%) | T11, T12 |
| AI requirement surge | Up | +29pp at entry, +39pp at mid-senior | **High** (consistent across measures) | T11, T12 |
| Senior role redefinition | AI + management evolving | Management vocabulary shifting (coaching/stakeholder replacing "management") | **Moderate** (Fightin' Words directional) | T10 |
| Large firm entry squeeze | Down | -13pp at 10K+ companies | **Moderate** (size data arshkon-only) | T13 |
| Entry-level SWE-specific decline | Yes | SWE -7pp vs control +21pp | **High** (opposite trends) | T16 |

### RQ2: Task and Requirement Migration

| Finding | Direction | Magnitude | Confidence | Source |
|---------|-----------|-----------|------------|--------|
| AI as dominant temporal distinguisher | Strong | z-scores -26 to -44 | **High** (Fightin' Words, 14-19 AI terms significant per comparison) | T10 |
| Entry-level vocabulary changed disproportionately | Yes | JSD 0.059 vs 0.038 overall (52% higher) | **High** | T11 |
| AI prevalence surge | Up | 19% -> 55% (broad); 3.6% -> 24% (narrow) | **High** (large effect, consistent) | T11, T14 |
| Ownership/autonomy language migration | Toward 2026 | ownership z=-11.9 (senior), autonomy z=-8.0 | **Moderate** (Fightin' Words directional) | T10 |
| Emerging terms dominated by AI tooling | Yes | agentic 6.7%, claude 4.9%, cursor 2.3% | **High** (clear pattern) | T11 |
| Traditional tech declining | Yes | waterfall, mongo disappearing | **Low** (small absolute counts) | T11 |

### RQ3: Employer-Worker Divergence

| Finding | Direction | Magnitude | Confidence | Source |
|---------|-----------|-----------|------------|--------|
| Employer AI requirements tripled | Up | 18% -> 55% (any AI) | **High** | T14 |
| AI-tool-specific surge | Up | 3% -> 26% (LLM/GPT/Copilot) | **High** (cleanest variable) | T14 |
| Employer-worker gap closed | Converged | ~40pp gap -> ~0pp | **Moderate** (worker benchmarks are survey-based) | T14 |
| Within-company AI surge | Yes | +33pp within-company, +1.5pp composition | **High** (not compositional) | T13 |

---

## 6. Key Tensions to Resolve

### 6.1 YOE decrease vs. scope increase

Entry-level YOE requirements dropped (mean 3.7 -> 2.0 years, T11) while organizational scope increased (cross-functional +10pp, ownership +8pp, T12). This is not credential inflation but **complexity inflation**: employers want fewer formal years but more organizational maturity. Possible mechanisms:
- AI tools reduce the experience needed for technical tasks, shifting emphasis to soft/organizational skills
- The definition of "entry-level" is expanding to include more organizational scope as a substitute for technical experience
- The YOE decrease may reflect formerly "mid-senior" roles being relabeled as "entry" with higher soft expectations (T12 hypothesis)

**Resolution needed:** The analysis should test whether the YOE decrease is concentrated in specific companies or broad, and whether it correlates with AI keyword presence.

### 6.2 Field-wide vs. SWE-specific

Scope inflation exists at all seniority levels (T12: field-wide), but the entry-level decline is SWE-specific (T16: control group shows opposite trend). These are not contradictory:
- Scope inflation is a profession-wide linguistic trend (longer descriptions, more organizational language everywhere)
- But the hiring pipeline restructuring (fewer entry-level openings, more AI requirements) is occupation-specific
- The claim should be: "SWE is experiencing both general scope inflation AND SWE-specific entry-level contraction"

### 6.3 Within-company vs. composition

The overall entry decline (-3.8pp) is weaker than the within-company decline (-9.3pp) because new 2026 companies partially offset with higher entry shares (+5.6pp composition effect, T13). The analysis should:
- Lead with within-company results as the better-identified finding
- Note that the composition effect dilutes the aggregate signal
- Use the 71-company micro-panel (3+ postings both periods) for company fixed-effects

### 6.4 AI convergence interpretation

The employer-worker AI gap closed (T14), but this could mean:
- Employers genuinely restructured jobs around AI (task-level change)
- Employers added AI keywords to postings without changing actual job content (signaling)
- AI tool adoption became universal enough that mentioning it is table stakes (institutionalization)

The interview artifacts (T17) are designed to probe which mechanism interviewees perceive.

---

## 7. Sensitivity Requirements

Every key finding in the analysis phase should be accompanied by the following sensitivity checks:

### 7.1 Aggregator exclusion

```sql
AND is_aggregator = false
```
Impact: Removes 6,056 asaniczka + 533 arshkon + 610 scraped SWE rows. T06 showed <1.5pp shift in seniority distributions after exclusion.

### 7.2 Company capping

Cap at 10 postings per company per period. T06 showed <1.5pp shift. Report capped results as sensitivity.

### 7.3 Regex-only SWE classification

```sql
AND swe_classification_tier = 'regex'
```
Drops from 30,283 to ~25,534 SWE rows (84%). Removes the 26% FP rate embedding_llm tier (T04).

### 7.4 Seniority variant table

For every seniority-stratified finding, report a table with at least:
- Variant A: seniority_patched (primary)
- Variant B: seniority_final (broken, for comparison)
- Variant C: seniority_native only (where available)

### 7.5 Description-length normalization

For keyword analyses, report:
- Binary indicator (any mention in posting) -- primary
- Rate per 1,000 characters -- sensitivity
- Show that the finding holds after normalization

### 7.6 Asaniczka exclusion

For any finding that uses asaniczka data, report the arshkon-vs-scraped result separately as the cleaner comparison.

### 7.7 "Jobs via Dice" exclusion

Exclude the 130 scraped rows from "Jobs via Dice" and the 53 from "Turing" in sensitivity.

---

## 8. Pipeline Fixes Needed Before Analysis

### 8.1 [BLOCKING] Stage 5 native_backfill bug

**File:** `preprocessing/scripts/stage5_classification.py`, `resolve_seniority_final()` (line ~624)

**Problem:** `infer_seniority_family()` does not align with `is_swe` classification, causing native_backfill to fire for only 1 of 30,283 SWE rows. 9,490 SWE rows with usable `seniority_native` labels are being ignored.

**Fix:** Align `infer_seniority_family()` with the `is_swe` classification path so that `family = 'swe'` is returned for all rows where `is_swe = true`.

**Impact of fix:** SWE unknown rate drops from 35% to 4%. Entry-level count increases from 345 to 1,273. The `seniority_patched` workaround replicates the expected behavior, but the pipeline should be fixed to make `seniority_final` correct natively.

**Workaround available:** `exploration/seniority_workaround.sql` -- use `seniority_patched` via SQL CTE.

### 8.2 [HIGH] Run seniority_llm

**Action:** Run Stage 10/11 with seniority classification task enabled for all SWE + SWE-adjacent + control rows.

**Impact:** Provides an independent third seniority signal. Resolves the direction disagreement (T09) by classifying the ~20% of rows that remain unknown even after the patched workaround.

### 8.3 [HIGH] Run swe_classification_llm

**Action:** Run Stage 10/11 with SWE classification task for embedding_llm tier rows.

**Impact:** Reduces the 26% FP rate in the embedding_llm tier (T04). Currently ~1,072 false positive SWE rows from hardware/manufacturing roles.

### 8.4 [MEDIUM] Add "Jobs via Dice" to aggregator list

**Action:** Add "Jobs via Dice" and "Turing" to Stage 2 aggregator patterns.

**Impact:** Removes ~183 scraped rows that are aggregator postings not currently flagged.

### 8.5 [LOW] Stage 12 validation

**Action:** Run Stage 12 end-to-end validation on a sample before full-batch LLM run.

**Impact:** Ensures the LLM pipeline produces valid `seniority_llm`, `swe_classification_llm`, and `ghost_assessment_llm` at scale.

---

## 9. File Index

### Reports
| Task | File | Key finding |
|------|------|-------------|
| T01 | `exploration/reports/T01.md` | Column coverage audit; seniority 45-71% known |
| T02 | `exploration/reports/T02.md` | Native backfill bug; seniority variants disagree |
| T03 | `exploration/reports/T03.md` | Missing data; entry n=89/124/132; associate/entry overlap |
| T04 | `exploration/reports/T04.md` | SWE FP ~5.5%; embedding_llm 26% FP |
| T05 | `exploration/reports/T05.md` | Description +50%; arshkon-scraped seniority comparable |
| T06 | `exploration/reports/T06.md` | Company concentration; capped sensitivity robust |
| T07 | `exploration/reports/T07.md` | Geographic r>0.93; same labor market regime |
| T08 | `exploration/reports/T08.md` | Entry share 21%->14%; desc +64%; YOE median 5->3 |
| T09 | `exploration/reports/T09.md` | 3/5 variants agree on decline; native-label cluster stronger |
| T10 | `exploration/reports/T10.md` | AI #1 distinguisher; ownership toward 2026; mgmt vocabulary evolving |
| T11 | `exploration/reports/T11.md` | Entry JSD 52% higher; AI 19%->55%; emerging terms = AI tooling |
| T12 | `exploration/reports/T12.md` | Scope inflation field-wide; YOE decreased; AI +29pp at entry |
| T13 | `exploration/reports/T13.md` | Within-company entry -9.3pp; AI surge within-company; large firms drive squeeze |
| T14 | `exploration/reports/T14.md` | AI tripled 18%->55%; AI-tools 3%->26%; gap closed ~40pp->0pp |
| T15 | `exploration/reports/T15.md` | Ghost flag too conservative; desc quality >99.8% ok |
| T16 | `exploration/reports/T16.md` | SWE junior decline is occupation-specific; control +21pp |
| T17 | `exploration/reports/T17.md` | 5 interview artifacts produced |

### Figures
Key figures for analysis phase (all in `figures/`):
- `T08/T08_fig3_seniority_5level.png` -- Seniority distributions
- `T09/T09_fig1_sensitivity_comparison.png` -- Seniority variant sensitivity
- `T10/ai_terms_heatmap.png` -- AI term z-scores across comparisons
- `T11/ai_prevalence_by_period_seniority.png` -- AI prevalence by period x seniority
- `T12/T12_scope_inflation_heatmap.png` -- Scope inflation across seniority
- `T13/T13_decomposition.png` -- Within-company vs composition decomposition
- `T14/t14_divergence_chart.png` -- Employer-worker divergence
- `T16/` -- Cross-occupation comparison charts (if generated)

### Tables
Key CSVs in `exploration/tables/`:
- `T11/ai_prevalence.csv` -- AI keyword rates by period x seniority
- `T11/emerging_overall.csv` -- Emerging terms
- `T13/` -- Company overlap and decomposition data

### Artifacts
Interview elicitation materials in `exploration/artifacts/`:
- `artifact1_inflated_junior_jds.md`
- `artifact2_paired_jds.md`
- `artifact3_junior_share_trend.png`
- `artifact4_senior_archetype.png`
- `artifact5_divergence_chart.png`
- `README.md`

### Workaround
- `exploration/seniority_workaround.sql` -- SQL CTE for seniority_patched (include in all seniority queries)
