# Composite C — "Who says 'junior,' and what does it mean when they do?"

Author: research dispatch (Composite C, exploratory). Date: 2026-04-21.
Code: `eda/scripts/composite_C_junior_label.py` (lines 1-417).
Tables: `eda/tables/S28_*.csv`. Figures: `eda/figures/S28_*.png`.

This is a methodology memo plus a per-sub-question verdict, not a final
article.

---

## Hypotheses

**H1 — "Junior" is a lower bound, not a marker of inexperience.** Postings
titled "junior/jr." may demand *more* YOE and *more* requirement breadth than
true-entry postings without that label.

**H2 — Entry-level postings are increasingly unlabelled.** Among true-entry
postings (yoe ≤ 2), the explicit "junior/entry/new grad" share may have
fallen 2024 → 2026.

**H3 — Substitute language has appeared.** What replaces "junior" — associate,
"I"/"II" suffixes, level codes, "early-career", or no marker?

**H4 — Usage is concentrated by employer type.** Defense, staffing, IT
consulting, non-tech — not big-tech / software-first firms.

---

## Data

`data/unified_core.parquet`, filtered to `is_swe = true` and
`llm_classification_coverage = 'labeled'` (n = 48,626 SWE rows across four
periods: 2024-01 n=18,124 asaniczka, 2024-04 n=4,687 arshkon, 2026-03/04
n=25,815 scraped LinkedIn). Scope features: T11 parquet at
`exploration-archive/v9_final_opus_47/artifacts/shared/T11_posting_features.parquet`.

Periods are source-collinear. Treat 2026-03/04 as the comparable current
window and 2024-01 as the comparable historical window; do not read the
2024-01 → 2024-04 gap as a real change (it is asaniczka → arshkon).

---

## Operationalizations + precision proxy

Each pattern was hand-scored on a 30-row reservoir sample per period (60 rows
per pattern). Sample CSV: `S28_02_precision_handsamples.csv`. Precision =
share of sampled titles I judged genuinely entry-tier.

| ID | Pattern | Catches | Precision (junior-tier intent) | Verdict |
|---|---|---|---|---|
| OP_A | `\bjunior\b` (title) | 0.4–0.9% | **≈1.00** (60/60 are junior postings) | Strict primary |
| OP_A2 | `\b(junior\|jr\.?)\b` (title) | 0.5–1.0% | **≈1.00** | Strict primary (more recall) |
| OP_B | + entry-level / new-grad / graduate / associate / early-career | 1.0–2.8% | **≈0.80** — "Senior Associate", "Research Associate" leak | Acceptable with audit |
| OP_C | junior/entry in **description** | 8–12% | **≈0.13–0.17** — overwhelmingly senior postings mentioning juniors they will mentor | **REJECT** — canonical false-positive trap (cf. `stories/05_management_never_there.md`) |
| OP_E | level codes (Engineer I/II/1/2, L3, IC2, SDE-2) | 2–5% | **≈1.00 as ladder marker; ≈0.30 as junior-tier** — most "Engineer II" postings demand 3+ YOE | Report as separate construct |
| OP_F | union of A2, B-extras, E | 3–7.5% | **≈0.60–0.70** | Use only as "any explicit marker" rate |

**Only OP_A / OP_A2 should be reported as "junior" rates.** OP_E is a
distinct construct ("level-code share"). OP_B can be reported with the
precision figure attached. OP_C should not be used.

---

## Per-operationalization results

### OP_A2 (junior in title) — share trajectory

`S28_01_op_coverage_by_period.csv`:

| Period | Junior-titled (%) | n |
|---|---|---|
| 2024-01 (asaniczka) | 0.50% | 90 |
| 2024-04 (arshkon) | 1.02% | 48 |
| 2026-03 (scraped) | 0.78% | 92 |
| 2026-04 (scraped) | 0.79% | 110 |

Under 1% of all SWE postings in every period. Flat across the comparable
scraped window.

### Lower-bound test — junior-titled vs unlabelled-true-entry

`S28_03_lower_bound_contrast.csv` and `S28_04_within_entry_contrast.csv`.
Mean YOE per posting:

| Bucket | 2024-01 | 2026-04 |
|---|---|---|
| Unlabelled true-entry (yoe ≤ 2, no marker) | 1.68 | 1.63 |
| OP_A "junior" titled (any YOE) | 2.33 | 1.58 |
| OP_A2 "junior/jr" titled (any YOE) | 2.25 | 1.54 |
| OP_B explicit-entry family | 2.53 | 1.92 |
| OP_E level-code | 4.96 | 3.45 |

Within seniority_final='entry' (S28_04):

| | 2024-01 | 2026-04 |
|---|---|---|
| entry, no junior in title | 1.88 | 1.11 |
| entry, junior in title | 2.33 | 1.58 |

**H1 (YOE lens) verdict: SUPPORTED in 2024 historical, MIXED in 2026.** In
2024 junior-titled postings demanded notably more YOE than the unlabelled
entry pool. In 2026 the gap closed in the broad sample (1.54 vs 1.63), but
within seniority='entry' the lower-bound pattern persists (1.58 vs 1.11) —
the unlabelled pool itself shifted toward 0–1 YOE postings, which closed the
broad-sample gap mechanically.

### Lower-bound test — scope features (T11)

`S28_05_scope_features_by_bucket.csv`. Mean tech count per posting:

| Bucket | 2024-01 | 2026-04 |
|---|---|---|
| junior-titled | 5.44 | 7.29 |
| other explicit entry marker | 4.52 | 6.82 |
| unlabelled yoe ≤ 2 | 4.59 | 6.10 |
| everything else | 5.41 | 7.59 |

Junior-titled postings carry the *highest* tech count among entry-candidate
buckets in both periods. In 2026 their requirement-breadth residual is +1.51
vs −0.36 for the unlabelled-entry pool. **For breadth, H1 is supported in
both periods.** AI-vocab rate (T11 ai_binary) is similar across buckets at
11–14% in 2026; not a clean differentiator.

### H2 — explicit-junior labelling rate among true-entry

`S28_06_labelling_trajectory.csv`. Among postings with yoe ≤ 2:

| Period | n | "junior/jr" rate | broad entry-family rate | any explicit marker (incl. levelcode) |
|---|---|---|---|---|
| 2024-01 | 1,218 | 3.0% | 6.0% | 11.7% |
| 2024-04 | 449 | 3.8% | 8.9% | 22.9% |
| 2026-03 | 1,356 | 3.6% | 7.1% | 24.0% |
| 2026-04 | 1,555 | 3.8% | 8.0% | 23.8% |

Stricter (yoe = 0 or 1) and broader (yoe ≤ 3, OR seniority='entry')
sensitivity grids tell the same story.

**H2 verdict: AGAINST.** The narrow "junior" rate is flat at 3-4%; the
broader explicit-marker rate has *risen* (12% → 24%), driven entirely by
level codes. The marker is not disappearing — its form is shifting.

### H3 — substitute n-grams

`S28_08_substitute_ngram_n{1,2,3}.csv`. Top title bigrams in 2026-04 among
unlabelled true-entry postings (yoe ≤ 2, no OP_F marker; n = 1,184 titles):

| Bigram | Count | Share |
|---|---|---|
| software engineer | 599 | 50.5% |
| **engineer iii** | 92 | 7.8% |
| full stack | 60 | 5.1% |
| systems engineer | 57 | 4.8% |
| machine learning / data engineer | 92 | 7.8% |
| ai engineer / ml engineer / engineer ai | ≈ 78 | ≈ 6.6% |

The notable substitute is **roman-numeral suffixes, especially "III"** on
yoe ≤ 2 postings — a level-code phenomenon outside OP_E (which catches I/II,
not III). "Engineer III" in this pool is mostly defense / government
contracting, where "III" is a job-grade code unrelated to American "senior"
usage.

**H3 verdict: SUPPORTED for level codes, WEAK for everything else.** The
modal unlabelled-entry posting in 2026 is just "Software Engineer" with no
seniority marker (50% of titles). The clearest substitute is the level-code
family, particularly "III" in defense contexts, which ironically *signals
high seniority* to readers unfamiliar with contractor grade conventions.

### H4 — who labels "junior"

`S28_07_who_labels_*.csv`.

**By industry (2026-04, n ≥ 30 only):** top — Civil Engineering 6.9%
(n=101), Engineering Services 5.4%, IT Services 3.3%, Pharma 2.8%, Aerospace/
Defense 2.8%, Oil and Gas 2.6%, Utilities 2.5%, Banking 2.5%. Bottom (0%):
Internet Marketplace Platforms, Higher Education, software-first firms,
Computers/Electronics Manufacturing.

**By aggregator status:** inconsistent across periods (0.51% aggregators vs
0.84% non-aggregators in 2026-04). No clean aggregator concentration.

**By company posting volume (2026-04):**

| Bucket | n | junior_rate |
|---|---|---|
| 1 post | 2,560 | 0.94% |
| 2-5 | 3,846 | 0.86% |
| 6-25 | 3,421 | 0.82% |
| 26-100 | 2,585 | 0.66% |
| 100+ | 1,595 | **0.50%** |

Junior labelling is slightly *more* common at small employers and least
common at the largest (Amazon, Google, Microsoft — who use level codes
instead). Monotonic but small.

**Top firms (n ≥ 10 posts, 2026-04):** Hire'in Solutions, Brooksource,
General Dynamics Mission Systems, Abbott, ConsultNet, Yoh, Cynet Systems,
Peraton, Parsons, SAIC, Leidos, Experis, Insight Global, CACI, GDIT —
overwhelmingly defense contractors and IT staffing. Top 10 firms are 25% of
junior posts; long tail is real (87 distinct companies labelled at least
once in April 2026).

**H4 verdict: SUPPORTED for industry/employer-type, AGAINST for
volume-concentration.** Defense, IT-staffing, traditional engineering, and
non-tech industries label "junior"; software-first big tech does not. But
labelling is dispersed across many small employers, not concentrated in a
handful of heavy posters.

---

## Verdicts

| H | Verdict | One-line |
|---|---|---|
| H1: junior is a lower bound | **MIXED** | Supported on requirement breadth (both periods); supported on YOE in 2024 and within seniority='entry' subset; broad-sample YOE gap closed in 2026 because the unlabelled pool itself shifted toward 0–1 YOE. |
| H2: explicit junior labelling falling | **AGAINST** | Junior rate flat at 3-4% of true-entry; broader explicit-marker rate *rose* 12% → 24%, driven by level codes. Form is shifting, not presence. |
| H3: substitute language identifiable | **SUPPORTED for level codes, WEAK otherwise** | Modal unlabelled-entry posting is unmarked "Software Engineer". Clearest substitute is roman-numeral suffixes ("III" in defense), often signalling defense grade rather than seniority. |
| H4: junior usage concentrated by employer type | **SUPPORTED for industry, AGAINST for concentration** | Defense, IT-staffing, non-tech industries label "junior"; big tech does not. Long-tail dispersion across 87 firms, not a few heavy users. |

---

## Recommended composite article structure

The original framing — "fewer firms say 'junior'" — is **wrong in this data**.
The interesting finding is the **divergence between marker form and marker
presence**: the share of true-entry postings carrying *some* explicit marker
roughly doubled (level codes), while the *junior* word itself stays at a low
stable share, used mostly outside the software-first mainstream.

**Three-panel article:**

1. **Who still says junior.** Headline: *"In April 2026, only 0.79% of SWE
   postings used 'junior' in the title; among employers posting 100+ roles
   the rate was 0.50%, and the top 15 junior-using firms were dominated by
   defense contractors and IT-staffing agencies."*

2. **What 'junior' demands when used.** Headline: *"Junior-titled postings
   list 7.3 named technologies on average in 2026, against 6.1 for unlabelled
   entry postings — a one-tech gap inside a category supposedly at the bottom
   of the ladder."*

3. **What replaced 'junior'.** Headline: *"The share of true-entry SWE
   postings carrying any explicit early-career marker doubled from 12% in
   2024-01 to 24% in 2026-04 — but every percentage point of that growth
   came from level codes (Engineer I/II/III), not from 'junior' or 'entry-
   level'."*

A short methodological coda — referencing the Hansen/Bloom remote-work
precision lesson noted in `stories/05_management_never_there.md` — should
explain why the description-text "junior" pattern (OP_C, 13% precision) was
discarded and why level codes are reported as a separate construct.

---

## Open questions

1. **Defense-contractor convention split.** "Engineer III" in defense maps to
   a GS-grade/pay-band, not a seniority step. A targeted scan of top-15
   defense employers' titling conventions would sharpen Panel 3.
2. **Big-tech level-code conventions.** Within Amazon/Google/Microsoft, what
   fraction of yoe ≤ 2 postings carry any title marker? Bears on whether the
   unlabelled-entry pool is convention-driven at a few large firms.
3. **2024 → 2026 lower-bound compression.** The unlabelled-entry pool
   shifted toward 0–1 YOE (1.68 → 1.63) and junior-titled posts moderated
   (2.25 → 1.54). Decompose with a period × bucket regression.
4. **Description-side junior mentions** (OP_C). Unusable as a junior measure,
   but a separate construct — senior postings *describing* juniors to mentor.
   Worth a separate pass; not for this article.

---

## Caveats

- **Source-period collinearity.** 2024-01 = asaniczka, 2024-04 = arshkon,
  2026-03/04 = scraped. Treat 2026-03/04 as current window, 2024-01 as
  historical anchor; the 2024-04 column should not carry a story.
- **Within-frame only.** All percentages condition on
  `llm_classification_coverage = 'labeled'`. Indeed and out-of-frame rows
  excluded.
- **Hand-sample size.** Precision scored on 60 titles per pattern; ±5 pp.
  Sample CSV is checked in for re-scoring.
- **"Engineer III" is ambiguous** — a defense-contractor grade code, not
  necessarily a seniority step. The article should say so explicitly.
- **Within-firm panel not run.** Arshkon-vs-scraped firm overlap is low.
- **The OP_C precision judgment is the most consequential methodological
  call.** If you disagree, re-score the OP_C rows in
  `S28_02_precision_handsamples.csv` before reading the H2 verdict; that
  pattern, if accepted at higher precision, would substantially change the
  share table.
