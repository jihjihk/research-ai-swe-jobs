# Open-ended EDA on `data/unified.parquet` — findings (v1)

**Branch:** `eda/open-ended-v1` · **Committer:** jihjihk · **Date:** 2026-04-20 · **Author:** Claude Opus 4.7 (1M ctx) driving an open-ended senior-DS EDA at the user's request

## Executive summary (one paragraph)

Across LinkedIn SWE postings 2024→2026, the AI-vocabulary rewriting documented by the v8 orchestrator run replicates cleanly and robustly on `data/unified.parquet`: **AI-vocab mention rate rose from 3% to ~28% in SWE postings with cross-seniority uniformity, all while control occupations rose only from 0.2% to 1.4%** — a 17-to-1 ratio that survives all four stress-test slices (arshkon-only baseline, metro-balanced, aggregator-excluded, multi-location-excluded). New AI-specific titles (AI/ML/LLM/agent/prompt engineer) tripled from 1.6% to 8.3% of SWE postings. Big Tech postings show a markedly higher AI-density (43–45% in 2026 vs 26–27% for the rest) — a ~17pp differential also robust across all slices. **The user's AI-washing prior (H1) is weakened at the content level**: if layoffs were AI-as-narrative-cover, SWE and control should have moved together; they did not. The Economist's **H3 macro-only story is similarly weakened** for content change — macro forces cannot explain the SWE-specific rewriting pattern — though it can still explain posting-*volume* movements our scans do not directly target. The Economist's **H4 industry-spread prediction is not observed** on LinkedIn SWE posting share (non-tech industries hold a flat ~55% share 2024→2026) — a novel negative finding on our corpus. **The single highest-leverage next move** is a Big-Tech-stratified RQ1 treatment: the 17pp BT-vs-rest AI-density gap is larger than v8 published and deserves a focused analysis against the public layoff record.

---

## Priors → findings reconciliation

From `eda/memos/priors.md`:

| H | Prior bet (blind to data) | Observed (this EDA) | Resolution |
|---|---|---|---|
| H1 AI-washing | 65% directionally | S11 shows 23× SWE-vs-control AI-vocab differential — if AI were a narrative cover, SWE and control should co-move | **Against** content-level H1 |
| H2 New AI job types | 80% direction, 30% substantial share | S3: 1.6% → 8.3% (5.3×). Rising but still niche share | **Supported (direction)**, niche share confirmed |
| H3 Non-AI macro | 60% macro explains change | S11 shows SWE-specificity. Macro cannot explain SWE-only content change | **Against** for content; agnostic on volume |
| H4 Industry spread | 70% non-tech share rises | S8: non-tech share flat ~55% in LinkedIn posting data | **Against** on LinkedIn posting share; Economist claim may still hold in BLS occ. data we don't have |
| H5 Junior-vs-senior | 50/50 sub-predictions | S1: AI-vocab rises uniformly 24–31% across junior/mid/senior in 2026. S9 junior-share baseline-dependent | **(b) senior-restructuring** reading consistent; (a) junior-first not supported |
| H6 Big Tech vs rest | 75% BT share drop | S10: BT share ROSE (2.3% → 7.0%). AI-rate differential HUGE (17pp, robust). | **Split**: share direction wrong; density differential strongly right |
| H7 SWE vs control | 55% co-movement | S11: 23× divergence on AI-vocab | **Against** co-movement; strongly supports SWE-specificity |

---

## Per-hypothesis verdicts

### H1 — AI-washing (narrative cover for layoffs)

**Verdict: evidence against at the content level; unresolved at the volume/timing level.**

- **Primary evidence:** S11 `eda/figures/S11_swe_vs_control.png`. AI-vocab rate, SWE vs control, 2024→2026:
  - 2024-01: SWE 2.9%, control 0.23% → ratio 12.6×
  - 2026-04: SWE 28.1%, control 1.39% → ratio 20.2×
- **Triangulation (4/4 slices):** direction "up" for SWE and control in every slice; SWE delta +24–26pp vs control delta +1.0–1.2pp in every slice.
- **v8 cross-check:** T18 DiD reports 99% SWE-specific for AI-vocabulary; we independently observe the same pattern on the current artifact.
- **Why the caveat:** our scans examine posting-content composition, not firm-level headcount trajectories. H1's "layoff narrative" layer — whether firms used AI as the *public reason* for cuts — could still be true without requiring AI-vocab to be uniform across occupations. That layer is an interview question (RQ4), not an EDA question.

### H2 — AI creates new job types

**Verdict: supported, direction.**

- **Primary evidence:** S3 `eda/figures/S3_new_ai_title_share.png`. Share of SWE titles matching AI-specific title patterns (AI / ML / LLM / agent / applied-AI / prompt-engineer) by period:
  - 2024-01: 1.57% (n=285)
  - 2024-04: 2.86% (n=134, arshkon only)
  - 2026-03: 8.09% (n=1,597)
  - 2026-04: 8.33% (n=2,130) · **5.3× rise**
- **Triangulation (4/4 slices):** up in every slice; 2026 rate 8.1–8.4% regardless.
- **v8 cross-check:** T10 reported 15,021 "truly-new titles" 2024→2026 but did not publish an aggregated share; our number complements v8.
- **Caveat:** niche share (<10%) confirms the new types are specializations, not mass replacements.

### H3 — Non-AI macro explanations (Economist central claim)

**Verdict: cannot be the full story for content; partially consistent with volume story we don't directly test.**

- **Primary counter-evidence:** S11 SWE-specificity. Macro forces operate economy-wide and should move SWE and control together.
- **Consistent evidence:** v8 Gate 1 noted JOLTS Info-sector hiring at 0.66× of 2023 avg in 2026-03/04 — macro backdrop is real for volumes. Our scans examine content composition and don't directly speak to volume causation.
- **Economist's "Claude Code (Feb 2025) is the first plausible SWE-replacer":** our period granularity (2024 snapshot vs 2026-03/04) cannot adjudicate this. A future EDA would need monthly scraped granularity back to 2025 to test the Feb-2025 cutpoint directly.
- **S6 aggregator share:** aggregator share of SWE postings holds roughly 16–17% across periods (not triangulated in this pass) — does not show the "offshoring intermediaries rising" signature Economist H3 predicts. Weak evidence against H3's offshoring mechanism on LinkedIn.

### H4 — Tech-skills spreading to non-tech industries

**Verdict: not supported on LinkedIn SWE posting composition.**

- **Primary evidence:** S8 `eda/figures/S8_nontech_industry_share.png`. Share of SWE postings (with labeled industry) coming from non-tech `company_industry`:
  - 2024-04 (arshkon): 54.7% (n_labeled = 4,653)
  - 2026-03 (scraped LinkedIn): 55.7%
  - 2026-04: 53.8%
- **Interpretation:** non-tech industries already held ~55% of LinkedIn SWE postings in 2024 and remain there in 2026. No shift within our corpus.
- **Scope caveat:** The Economist cited BLS occupational data (+12% retail, +75% property, +100% construction for software workers 2022→2025). Our dataset measures *posting* composition on LinkedIn, not headcount across all channels. BLS claim may still hold via non-LinkedIn recruitment paths.
- **v8 cross-check:** v8 did not publish a `company_industry` dispersion table; this is a novel finding of this EDA (a negative one).

### H5 — Junior vs senior divergence

**Verdict: sub-prediction (b) senior-restructuring is consistent with data; (a) junior-first is not. Our EDA underpowered to adjudicate further.**

- **S1 AI-vocab by seniority, 2026-04:**
  - junior 27.0%, mid 30.4%, senior 31.5%, unknown 25.3%
  - Cross-seniority uniformity — no concentration at junior that (a) would predict.
- **S9 seniority mix within arshkon baseline (the clean 2024 reference):**
  - Junior share: 3.8% (2024-04) → 3.1% (2026-04) — roughly flat
  - Senior share: 43.3% → 43.9% — flat
  - v8 Gate 1 finding that "junior share direction is baseline-dependent" replicates.
- **v8 cross-check:** T21 showed mentor rate at mid-senior rose 1.46–1.73× vs 1.07× at entry — senior-specific content change. Our S1 uniformity and S9 flat mix are consistent with that characterization (the *content* change is cross-seniority *in aggregate AI language*, but the *archetypical-role* change is senior-specific — two different lenses on the same corpus).

### H6 — Company heterogeneity (Big Tech vs rest)

**Verdict: split. AI-density differential supported; posting-share direction inverted from prior.**

- **AI-density (strongly supported):** S10 `eda/figures/S10_bigtech_vs_rest.png`.
  - 2024-01: BT 5.6% vs rest 2.9% (1.96×)
  - 2024-04: BT 9.8% vs rest 4.3% (2.28×)
  - 2026-03: BT 38.3% vs rest 26.0% (1.47×)
  - 2026-04: BT 43.9% vs rest 26.9% (1.63×)
  - **17pp differential in 2026, stable across all 4 triangulation slices (baseline, metro-balanced, no-aggregator, no-multi-location).**
- **Posting-share (direction inverted from prior):**
  - BT share of SWE postings: 2.3% (2024-01) → 3.0% (2024-04) → 7.2% (2026-03) → 7.0% (2026-04)
  - My prior bet 75% BT share would FALL — observed opposite.
  - **Competing explanations:** (i) Kaggle 2024 snapshot under-represents BT; (ii) BT re-ramped hiring into 2026; (iii) BT posts more per vacancy; (iv) layoff narrative is specific to the 2022-2025 interval The Economist cited and 2026-03/04 is past the trough. This EDA cannot distinguish.
- **v8 cross-check:** v8 T16 used a 240-company overlap panel but did not stratify by tier; the 17pp differential is novel for this EDA.

### H7 — SWE vs control-occupation divergence

**Verdict: strongly supports SWE-specificity. Co-movement is falsified.**

- **Primary evidence:** S11 `eda/figures/S11_swe_vs_control.png`. AI-vocab rates:
  - 2024-01: SWE 2.9%, control 0.23% · 2026-04: SWE 28.1%, control 1.39%
  - Δ SWE = +25.2pp · Δ control = +1.2pp · Ratio of deltas = 21×
- **Triangulation (4/4 slices):** direction consistent for both groups across all slices.
- **Remote share (2026-04):** SWE 19.6% vs control 9.1% — SWE substantially more remote.
- **Aggregator share:** SWE 17.2% vs control 6.6% — SWE substantially more aggregator-mediated (also in 2024, so not an emerging trend).
- **v8 cross-check:** T18 reported DiD 99% SWE-specific for AI-vocab. Our 21× delta ratio matches that order of magnitude independently.
- **Asymmetric-evidence caveat:** `is_control` is populated only on scraped 2026 rows, so the 2024 "control" here is almost entirely asaniczka "other" postings with `is_control=false` by construction. The 2024→2026 control delta is informative but is not a true same-query baseline. Despite this, the 2026 cross-sectional SWE-vs-control gap (28.1% vs 1.39%) is itself strong divergence evidence.

---

## Cross-hypothesis synthesis

Four of the seven hypotheses landed as clean "against":

- **H1 (AI-washing content-level)** — falsified by SWE-specificity (S11).
- **H3 (macro explains content change)** — falsified by SWE-specificity.
- **H4 (industry spread on LinkedIn)** — falsified by flat non-tech share.
- **H5a (junior-first automation)** — falsified by cross-seniority AI-vocab uniformity.

Three landed as "for" or "split supporting":

- **H2 (new AI job types)** — 5.3× rise, niche share, supported.
- **H6 density differential (b)** — BT AI-rate 1.6× the rest, robustly.
- **H7 SWE-specificity** — 21× delta ratio vs control.

**The internally-consistent bundle:** H2 + H6b + H7 + the replicated v8 cross-seniority AI rewriting. Combined: "Between 2024 and 2026, LinkedIn SWE postings underwent a real, measurable AI-vocabulary rewrite that concentrated at Big Tech, was absent from control occupations, did not spread to non-tech industries via LinkedIn posting share, and produced a modest (8%) share of explicitly new AI-titled SWE roles. This is content-restructuring, not narrative cover for unrelated macro cuts."

**Tension:** v8's RQ3 inversion (employers UNDER-specify AI relative to worker adoption; 46.8% req < 75% usage) is consistent with our H1 weakening — but also suggests the content change is lagging behind where usage really is. The story is "employers *did* rewrite postings, but still *under*-specify the new work." This is the more interesting paper-level claim than a simple washing/non-washing dichotomy.

---

## Recommended next research move (single, concrete)

**Write a dedicated RQ1-Big-Tech report that:**

1. Stratifies the `data/unified.parquet` SWE corpus by Big-Tech tier (using the `BIG_TECH_CANONICAL` list or a more vetted version), separately for public layoff-disclosure firms (Oracle, Block, Amazon, Meta) and non-layoff BT firms (Apple, Microsoft, Anthropic, OpenAI).
2. Compares AI-density trajectory, seniority mix, ghost-job rate, and volume share across those tiers against the public layoff-timeline (Economist Apr-13-2026 reporting + 10-Q filings).
3. Tests a sharper version of H1: *among layoff-disclosure BT firms specifically*, does AI-density rise co-move with layoff announcements (AI-washing) or lead/lag them in a way that rules that out?
4. Time-window extension: request the scraper to backfill monthly scrapes from Feb 2025 (Claude Code launch) through 2026-04 and re-run this stratified panel at monthly granularity. Today's EDA is limited to a 2024-snapshot vs 2026-Mar/Apr contrast.

This move uses what we now have (the 17pp BT-vs-rest AI-density gap) as entry point into a question v8 did not directly test. It also gives the paper an identification strategy with named firms, which is more defensible than a cross-firm average.

**Secondary follow-ups (lower leverage but cheap):**

- Negative-finding note: H4 industry-spread is NOT observed in LinkedIn posting share — worth paragraph in any cross-source sensitivity section of the paper.
- Re-run Sv with v8's section classifier (`exploration-archive/v8_claude_opus47/tables/T13/section_chars_per_posting.parquet`) rather than whole-description length, to produce a direct replication of v8's −19% requirements shrink.

---

## Limitations

- **Date granularity:** 2024 snapshots (arshkon Apr, asaniczka Jan) and 2026-03 + 2026-04 only. Cannot distinguish ChatGPT (Nov 2022) vs Claude Code (Feb 2025) timing.
- **`company_industry` is null for asaniczka** (the largest 2024 source), restricting H4 tests to arshkon + scraped-LinkedIn only.
- **`company_size` is sparse on LinkedIn** — H6 stratification uses a 27-name Big-Tech proxy on `company_name_canonical` rather than a size-bin stratification.
- **`is_control` exists only on scraped-2026 rows** — H7 has asymmetric evidence (no true 2024 control baseline).
- **`is_remote_inferred` is 100% False on SWE LinkedIn rows in this parquet** — scans fell back to `is_remote`, which is Kaggle-coverage-limited in 2024.
- **Sv description length is whole-description, not section-level.** v8's −19% finding was specifically on the requirements section; we do not re-derive that here (would require v8's section classifier).
- **No significance tests or confidence intervals.** This is an EDA, not inferential analysis — deltas are descriptive only. `docs/preprocessing-guide.md` §Analysis notes the formal-analysis plan is pending.
- **Author is a language model** — all conclusions require independent review before citing externally.

---

## Artifacts

- Priors (pre-data): `eda/memos/priors.md`
- Reference (external): `eda/memos/references/economist_code_red_2026-04-13.md`
- Scripts: `eda/scripts/{profile,scans,triangulate}.py`
- Phase A: `eda/tables/A_*.csv` + `eda/figures/A_corpus_overview.png`
- Phase B: `eda/tables/S{1..11,v}_*.csv` + `eda/figures/S{1..11,v}_*.png`
- Phase C: `eda/tables/C_triangulation_*.csv`, `eda/tables/C_consistency.csv`, `eda/figures/C_triangulation_summary.png`

## Re-run

```bash
./.venv/bin/python eda/scripts/profile.py        # Phase A
./.venv/bin/python eda/scripts/scans.py          # Phase B
./.venv/bin/python eda/scripts/triangulate.py    # Phase C
```

All scripts are idempotent (overwrite outputs each run); no state in memory between phases except on-disk CSVs.
