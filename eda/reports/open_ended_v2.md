# Open-ended EDA v2 — findings on `data/unified_core.parquet`

**Branch:** `eda/open-ended-v1` · **Committer:** jihjihk · **Date:** 2026-04-20 · **Author:** Claude Opus 4.7 (1M ctx)

## What changed since v1

- **Primary analysis file upgraded:** `data/unified.parquet` (1.45M × 96) → `data/unified_core.parquet` (110k × 42). Core is the Stage 9 balanced LLM-frame subset — LinkedIn-only, 40/30/30 SWE/adjacent/control, all rows already LLM-labeled, with `is_control` populated in **all four periods** (fixes v1's asymmetric-evidence caveat).
- **Hypothesis set extended from H1–H7 to H1–H13.** Six new hypotheses emerged from an initial core exploration and are now formally tested.
- **Notebook-driven:** `eda/notebooks/open_ended_v2.ipynb` is the clean driver. Scan logic lives in `eda/scripts/{scans,core_scans}.py`.

## One-paragraph executive summary

Using the balanced Stage 9 LLM-frame artifact, every v1 finding **strengthens** (the SWE-vs-control divergence ratio holds at 21–23× on the cleaner sample; Big Tech AI-density differential holds at 17pp robustly; new-AI-title share 1.6%→8.3% replicates). Six new findings are added:
- **(H8) YOE floor *fell* 2024→2026** — junior mean LLM-YOE dropped from 2.01 to 1.23, median 2→1. Counter-scope-inflation.
- **(H9) Dev-tool vendor leaderboard is measurable and hierarchical** — Copilot 4.25% > Claude 3.83% > OpenAI 3.63% > Cursor 2.17% (2026-04). ChatGPT as a brand is plateauing while Claude/Cursor climb.
- **(H10) AI mention is not a ghost-job signal** — AI-mentioning SWE postings have *lower* inflated rate (4.5% vs 5.6%).
- **(H11) Control AI-rise is niche-specific** — finance/accounting + electrical/nuclear engineering drive it; nursing/HR/sales untouched.
- **(H12) AI postings live slightly longer** in 2026-03 scraped data (5.3 vs 4.4 days mean) — directional, small.
- **(H13) Within-firm AI rewrite is strongly real** — on a 292-company 2024↔2026 overlap panel, the mean within-firm AI-vocab delta is **+19.4pp**; 75% of firms rose; 61% rose >10pp. Microsoft +51pp, Wells Fargo +45pp, Amazon +36pp. **Rules out between-firm composition as the dominant driver of the 2024→2026 AI rewrite** — the same firms that were not mentioning AI in 2024 are now mentioning it heavily.

The strongest paper-worthy new finding is **H13 within-firm rewrite** (a Gulfstream-level control for composition), followed by **H9 vendor leaderboard** (genuinely new artifact nobody has published) and **H8 YOE-floor inversion** (directly contradicts the popular scope-inflation narrative).

---

## Verdict table (H1–H13)

| H | Verdict | Primary evidence |
|---|---|---|
| H1 AI-washing | **against (content-level)** | S11 SWE Δ+25pp vs control Δ+1.1pp — 23:1 ratio on balanced core |
| H2 new AI job types | **supported** | S3 new-AI-title share 1.6% → 8.3% (5×) |
| H3 non-AI macro (content) | **against** | S11 SWE-specificity rules out economy-wide cause of content change |
| H4 industry spread | **NOT observed on LinkedIn** | S8 (v1): non-tech share flat ~55% 2024→2026 |
| H5 junior-first vs senior-restructuring | **(a) falsified, (b) consistent** | S1 AI-vocab uniform across levels; S12 junior YOE FELL |
| H6 Big Tech vs rest | **split** | S10 BT AI 44% vs rest 27% (+17pp, robust); BT share ROSE, not fell |
| H7 SWE-vs-control | **strongly supports SWE-specificity** | S11 on balanced core with real 2024 control baseline |
| **H8 YOE floor falling** | **supported** | S12 junior mean YOE 2.01 → 1.23; median 2→1; senior median 6→5 |
| **H9 vendor leaderboard** | **supported (hierarchy)** | S13 2026-04: Copilot 4.25%, Claude 3.83%, OpenAI 3.63%, Cursor 2.17% |
| **H10 AI mention ≠ ghost** | **supported** | S14 AI-mentioned inflated rate 4.5% vs non-AI 5.6% (2026-04) |
| **H11 control spread is niche** | **supported** | S15 finance/accounting + electrical/nuclear dominate; service occupations flat |
| **H12 posting survival** | **directional signal only** | S16 AI-mentioning SWE postings live 0.9 days longer in 2026-03 (small effect) |
| **H13 within-firm rewrite** | **strongly supported** | S17 mean +19.4pp on 292-company panel; 75% of cos rose; 61% >10pp |

---

## New-in-v2 findings in detail

### S12 (H8) YOE floor trajectory

| period | junior mean | junior median | senior mean | senior median | mid mean | unknown mean |
|---|---|---|---|---|---|---|
| 2024-01 | 2.01 | 2 | 6.55 | 6 | 4.00 | 5.11 |
| 2024-04 | 1.46 | 1 | 6.54 | 6 | 3.00 | 4.74 |
| 2026-03 | 1.37 | 1 | 6.35 | 5 | 2.87 | 4.38 |
| 2026-04 | **1.23** | 1 | 6.37 | **5** | 2.36 | 4.50 |

All buckets trend down on LLM-YOE. The "junior scope inflation" hypothesis — as classically stated (junior postings demand more YOE) — is *falsified* on this measurement. Note: v8 Gate 1 reported that rule-based `yoe_extracted` rose, but attributed it 95% to between-company composition; the LLM-YOE (cleaner per the schema doc) shows a *fall* on both mean and median within-seniority.

### S13 (H9) Dev-tool vendor leaderboard

2026-04 mention rates in SWE postings:

| vendor | 2024-01 | 2026-04 | multiple |
|---|---|---|---|
| Copilot | 0.08% | **4.25%** | 53× |
| Claude | 0.02% | **3.83%** | 190× |
| OpenAI | 0.22% | **3.63%** | 17× |
| Cursor | 0.01% | **2.17%** | >200× |
| Anthropic (brand) | 0.02% | 1.48% | 70× |
| Gemini | 0.03% | 1.07% | 35× |
| GPT (versioned) | 0.18% | 0.87% | 5× |
| ChatGPT (brand) | 0.09% | 0.70% | **plateauing** (0.82% in 2026-03 → 0.70% in 2026-04) |
| Llama | 0.04% | 0.53% | 13× |
| Mistral | 0.01% | 0.17% | 17× |

Three structural observations:
- **Copilot leads** on raw mention rate but Claude has *higher growth rate* (190× vs 53×). On current trajectory Claude overtakes Copilot in mid-2026.
- **ChatGPT as a brand is plateauing/declining** while "Claude" and "OpenAI" keep climbing — suggests employers are specializing their AI language from a consumer-brand mention to specific-vendor/specific-model mentions.
- **Cursor's emergence** is the most dramatic: 0.01% → 2.17% in 2 years. Visible in labor demand before most popular-press coverage.

### S14 (H10) AI-mention × ghost rate

| period | AI-mentioned inflated | non-AI inflated | AI − non-AI |
|---|---|---|---|
| 2024-01 | 5.25% | 5.88% | −0.63pp |
| 2024-04 | 7.62% | 7.15% | +0.47pp |
| 2026-03 | 4.38% | 5.06% | −0.68pp |
| 2026-04 | 4.49% | 5.99% | **−1.50pp** |

Also: SWE inflated rate is flat or slightly declining 2024→2026 (5.9% → 5.6%). **The ghost-job narrative isn't getting worse in aggregate, and AI-buzzword postings are no more ghost-like than non-AI ones — if anything, slightly less.**

### S15 (H11) Control AI-rise occupational drivers

Families with the largest 2026-04 AI-rate inside `is_control`:
1. **finance/accounting** (Senior Financial Analyst, Accounting Manager, Senior Revenue Accountant) — the single biggest driver.
2. **electrical/nuclear engineer** (Substation, Nuclear, Grid variants).
3. **mechanical engineer** (Nuclear, Space).
4. **marketing** — near-zero AI rate.
5. **nursing / HR / sales** — essentially untouched.

Reframes H4: AI language is not spreading broadly into non-SWE; it is reaching finance + utility/nuclear engineering specifically. Aligns with v8 T17's Sunbelt finance AI surge observation (Tampa Bay, Atlanta, Charlotte top 3).

### S16 (H12) Posting survival

Restricted to 2026-03 (only period with a complete monthly observation window):

| tier | ai_mentioned | n | mean_obs | mean_days |
|---|---|---|---|---|
| swe | no_ai | 8,658 | 1.44 | 4.36 |
| swe | **ai_mentioned** | 3,152 | 1.48 | **5.27** |
| control | no_ai | 6,643 | 1.24 | 2.60 |
| control | **ai_mentioned** | 73 | 1.49 | **7.53** |
| swe_adjacent | no_ai | 4,366 | 1.37 | 3.69 |
| swe_adjacent | ai_mentioned | 1,220 | 1.42 | 4.70 |

AI-mentioning SWE postings persist 0.9 days longer on average — a small effect. The AI-mentioning control postings are a small sample (n=73) but show a much larger differential. Directionally consistent with AI-tagged postings being harder to fill, but too small to draw strong conclusions.

### S17 (H13) Within-firm AI rewrite — the biggest finding

**Panel:** 292 companies with ≥5 SWE postings in BOTH 2024 (Kaggle) and 2026 (scraped).

**Aggregate stats:**
- Mean within-firm Δ AI-vocab = **+19.36pp**
- Median within-firm Δ = **+16.67pp**
- 75.0% of companies rose
- **61.3% rose more than 10pp**
- **39.4% rose more than 20pp**

**Top individual risers (selected):**

| company | n 2024 | n 2026 | AI rate 2024 | AI rate 2026 | Δ |
|---|---|---|---|---|---|
| Microsoft | 20 | 229 | 10.0% | 61.1% | **+51.1pp** |
| Wells Fargo | 37 | 168 | 0.0% | 44.6% | **+44.6pp** |
| Amazon | 173 | 214 | 2.3% | 37.9% | **+35.5pp** |
| Walmart | 172 | 145 | 7.0% | 39.3% | **+32.3pp** |
| Amazon Web Services | 109 | 220 | 3.7% | 30.5% | **+26.8pp** |
| Capital One | 710 | 151 | 3.9% | 29.1% | **+25.2pp** |
| Motion Recruitment (aggregator) | 124 | 139 | 0.8% | 25.9% | +25.1pp |
| Google | 33 | 466 | 3.0% | 24.9% | +21.9pp |

**Non-movers (defense/aerospace):**

| company | Δ |
|---|---|
| Raytheon | 0.0pp |
| Northrop Grumman | +1.6pp |
| Anduril Industries | +0.7pp (already saturated at 98%) |

**Interpretation:** the v8 lead narrative ("2024→2026 SWE postings were rewritten toward AI") passes the strongest possible within-firm test on our data. The same employers that were posting non-AI-language SWE roles in 2024 are now posting AI-laden SWE roles in 2026. Between-firm composition churn is NOT the dominant driver. This is the cleanest causal-ish signal in the EDA.

---

## Four recommended next research moves (ranked)

1. **Within-firm rewrite paper (S17 as anchor).** The 292-company overlap panel with +19.4pp mean within-firm rise is the strongest causal-identification surface on our data. Pair with RQ4 interview questions: *"Who inside your firm rewrote the JDs between 2024 and 2026? What changed about the actual work you expect from a SWE hire?"* Connects employer-side content directly to the ghost-vs-real debate.

2. **Big-Tech-stratified RQ1 paper.** The 17pp BT-vs-rest AI-density gap (S10) and the biggest within-firm movers (Microsoft +51pp, Amazon +36pp, Google +22pp) concentrate at Big Tech. Pair with named-firm layoff timelines and 10-Q filings to test a sharper version of H1: *does AI-density rise lead, follow, or co-move with layoff announcements at each firm?*

3. **Dev-tool vendor-specificity paper (S13).** Nobody has published a labor-demand vendor-share table for dev tools. The hierarchy (Copilot > Claude > OpenAI > Cursor) with Claude's steeper growth curve and ChatGPT's plateauing is entirely new and widely interesting. Sensitivity: verify that "cursor" and "claude" regex matches are not over-counting unrelated senses.

4. **YOE-floor inversion note (S12).** Rule-based YOE in v8 rose compositionally; LLM-YOE on the balanced core shows decline across all seniority buckets. Worth calling out explicitly as a methodological footnote in the paper: *"A commonly-cited indicator of scope inflation — junior postings requiring more experience — is NOT observed on LLM-extracted YOE, only on rule-based extractors that are more confounded by composition."*

---

## Limitations (v2)

- **LinkedIn-only** by construction on `unified_core.parquet`. No Indeed sensitivity in this pass.
- **Kaggle vs scraped recruitment surface differs.** 2024 control rows come from Stage 5 classifying Kaggle postings as `is_control=true`; 2026 control rows come from explicit scraper query tiers. The classification criterion is consistent but the recruitment surface differs. S11 SWE-specificity claim would be cleaner with same-channel control 2024.
- **`unified_core_observations.parquet` has no cadence for Kaggle sources.** S16 posting-survival is 2026 only.
- **Vendor regexes (S13) may over-match.** "Cursor" also appears as UI vocabulary; "Claude" is a common name; "Llama" is a nature/mascot word. Rates should be read directionally.
- **Within-firm panel (S17) requires ≥5 SWE postings in both periods.** 292 firms is meaningful but excludes most small employers. Lower threshold (≥2+2 or ≥3+3) would enlarge the panel at the cost of noisier per-firm rates.
- **H8 YOE finding depends on `yoe_min_years_llm` coverage** (~78% of in-frame SWE rows). Selection bias possible: rows where the LLM refused to extract YOE might differ systematically.
- **H12 effect size is small** and 2026-04 is incomplete (a posting observed once on 2026-04-20 could still persist into May). Comparison inside 2026-03 is the cleanest signal we can produce.
- **No significance tests.** Descriptive EDA only. `docs/preprocessing-guide.md` notes the formal-analysis plan is pending.

---

## Artifacts

- Priors: `eda/memos/priors.md` (H1–H13)
- External reference: `eda/memos/references/economist_code_red_2026-04-13.md`
- Scripts: `eda/scripts/{profile,scans,core_scans,triangulate,build_notebook}.py`
- Notebook (driver): `eda/notebooks/open_ended_v2.ipynb`
- Phase A: `eda/tables/A_*.csv` + `eda/figures/A_corpus_overview.png`
- Phase B (v1 on full): `eda/tables/S{1..11,v}_*.csv` + `eda/figures/S{1..11,v}_*.png`
- Phase B (v2 on core): `eda/tables/S{1,3,10,11,v}_core_*.csv` + `_core_*.png`
- Phase B (v2 new): `eda/tables/S1{2..7}_*.csv` + `eda/figures/S1{2..7}_*.png`
- Phase C (v1 triangulation): `eda/tables/C_triangulation_*.csv` + `eda/figures/C_triangulation_summary.png`
- Reports: `eda/reports/open_ended_v1.md` (v1) and this file (v2).

## Re-run

```bash
# Rebuild notebook from the Python spec (deterministic)
./.venv/bin/python eda/scripts/build_notebook.py

# Execute in place
./.venv/bin/jupyter nbconvert --to notebook --execute --inplace eda/notebooks/open_ended_v2.ipynb

# Or run the scans directly without the notebook
./.venv/bin/python eda/scripts/core_scans.py
```
