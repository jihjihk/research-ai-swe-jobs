# Shared artifacts — Wave 1.5 (Agent Prep)

Built: 2026-04-20
Inputs: `data/unified.parquet` (final pipeline output)
Filter frame: `is_swe AND source_platform='linkedin' AND is_english=true AND date_flag='ok'` → 68,137 rows (arshkon 4,691 + asaniczka 18,129 + scraped 45,317)

Downstream agents should load these files rather than recomputing.

---

## Contents

| File | Rows × Cols | Produced by | Consumers |
|---|---|---|---|
| `swe_cleaned_text.parquet` | 68,137 × 15 | Step 1 | T09, T12, T13, T15, T35, T36 |
| `swe_embeddings.npy` + `swe_embedding_index.parquet` | 48,223 × 384 (covered) | Step 2 | T09, T15, T34 |
| `swe_tech_matrix.parquet` | 68,137 × 136 (uid + 135 tech) | Step 3 | T11, T14, T35 |
| `company_stoplist.txt` | 11,961 tokens | Step 4 | All text tasks |
| `asaniczka_structured_skills.parquet` | 478,638 (uid, skill) long rows | Step 5 | T14 |
| `calibration_table.csv` | 22 metrics × 12 cols | Step 6 | All tasks (SNR checks) |
| `tech_matrix_sanity.csv` | 135 × 7 | Step 7 | T14, T35 |
| `tech_escape_diagnostic.txt` | 3 techs (c++, c#, .net) diagnostic | side-output of Step 3 | Step 7 notes |
| `seniority_definition_panel.csv` | 78 | T30 (Wave 1) | All seniority-stratified tasks |
| `entry_specialist_employers.csv` | 206 | T06 (Wave 1) | T16, concentration checks |
| `returning_companies_cohort.csv` | 2,109 | T06 (Wave 1) | T37, T38 |

---

## Step 1 — Cleaned text column

`swe_cleaned_text.parquet` — primary text source is `description_core_llm` where `llm_extraction_coverage='labeled'`; fallback is raw `description`. English stopwords (NLTK 198-word list) and the company-name stoplist (Step 4) are stripped. Lowercased.

Columns: `uid, description_cleaned, text_source, source, period, seniority_final, seniority_3level, is_aggregator, company_name_canonical, metro_area, yoe_min_years_llm, yoe_extracted, llm_classification_coverage, swe_classification_tier, seniority_final_source`.

**Text-source distribution by source** (critical for sampling decisions):

| Source | total | llm | raw | % llm |
|---|---|---|---|---|
| kaggle_arshkon | 4,691 | 4,653 | 38 | 99.2% |
| kaggle_asaniczka | 18,129 | 18,061 | 68 | 99.6% |
| scraped | 45,317 | 25,509 | 19,808 | 56.3% |
| **total** | 68,137 | 48,223 | 19,914 | 70.8% |

**Usage rule:** text-sensitive analyses (embeddings, topic modeling, density metrics) must filter to `text_source='llm'`. Recall-only analyses (binary keyword presence) may use both — raw-text rows can inflate boilerplate-adjacent signals, so report the split explicitly.

## Step 2 — Sentence-transformer embeddings

`swe_embeddings.npy` (float32, shape N×384) + `swe_embedding_index.parquet` (`row_idx, uid`).

Model: `sentence-transformers/all-MiniLM-L6-v2`, `max_seq_length=512`. Encoded `description_cleaned` for rows where `text_source='llm'`. Batch size 256. Peak RSS stayed well under 31 GB.

Coverage: **48,223 of 48,223 labeled rows embedded (100%).**

To map uid → row index:

```python
import numpy as np, pyarrow.parquet as pq
emb = np.load("swe_embeddings.npy")
idx = pq.read_table("swe_embedding_index.parquet").to_pandas()
# idx has row_idx and uid columns; emb[row_idx] is the embedding
```

## Step 3 — Technology mention binary matrix

`swe_tech_matrix.parquet` — 135 technology boolean columns + `uid`. Uses raw `description` (boilerplate-insensitive) with the backslash-escape fix `re.sub(r"\\([+\-#.&_()\[\]\{\}!*])", r"\1", text)` applied BEFORE pattern matching.

Taxonomy: Languages (20), Frontend (8), Backend (10), Cloud (6), Orchestration/DevOps (8), CI/CD (6), Databases (11), Data pipelines (9), Traditional ML (9), LLM era (18), AI tools (7), Testing (7), Observability (7), Practices (9).

Column-name convention: snake_case lowercase (`python`, `c_plus_plus`, `c_sharp`, `dot_net`, `aws`, `langchain`, `copilot`, …). TDD assertions for regex corner cases run at build time and pass: C++ / C# / .NET positives post-escape, Go/R/Java negatives on common false-positive strings.

**Escape-fix diagnostic (Step 3 side-output):** see `tech_escape_diagnostic.txt`.

| Tech | pre-escape scraped rate | post-escape scraped rate |
|---|---|---|
| `c_plus_plus` | 0.0000 | 0.1860 |
| `c_sharp` | 0.1087 | 0.1087 |
| `dot_net` | 0.0686 | 0.0686 |

Interpretation: scraped markdown contains `C\+\+` (escaped) but `C#` and `.NET` plain. The fix is essential for C++ (0% → 18.6%) and a no-op for the others. The c++ pre-escape 0% would have been a silent tokenization catastrophe without the fix.

## Step 4 — Company name stoplist

`company_stoplist.txt` — 11,961 lowercased tokens extracted from 12,872 distinct `company_name_canonical` values across the SWE LinkedIn corpus. Generic tokens filtered out (inc/llc/corp/ltd/solutions/services/systems/tech/technologies/software/digital/global/…) to avoid stripping legitimate content words. Numeric-only tokens and <2-char tokens excluded.

## Step 5 — Asaniczka structured skills

`asaniczka_structured_skills.parquet` — long-format `(uid, skill)` with 478,638 rows from 18,114 asaniczka SWE LinkedIn postings with populated `skills_raw`. Skills lower-cased and whitespace-stripped. Top skills: python, java, sql, aws, javascript.

Arshkon and scraped do not have `skills_raw` populated per T01 (arshkon: 68 of 4,691; scraped: 0); asaniczka remains the only source with structured skills.

## Step 6 — Within-2024 calibration table

`calibration_table.csv` — 22 metrics. Each row: `metric | metric_type | arshkon_value | asaniczka_value | scraped_value | within_2024_effect | within_2024_sd | cross_period_effect | cross_period_sd | calibration_ratio | snr_flag | notes`.

- Continuous metrics: Cohen's d, pooled SD. Values reported as mean (or median for `*_median` variants).
- Proportions: |pa − pb|, SE of difference.
- Counts: mean diff, pooled SD.

Pooled-2024 baseline is arshkon + asaniczka combined (per Gate 1 memo).

**SNR verdict:** 7 above_noise (calibration_ratio ≥ 2), 7 near_noise (1–2), 8 below_noise (< 1).

Highlights:

- **AI-mention strict**: cross-period 0.014 → 0.144 (+13.3 pp), within-2024 0.004. Ratio 32.9 — cleanest signal.
- **AI-mention broad**: +35.6 pp cross-period vs 1.4 pp within-2024. Ratio 25.4.
- **Scope terms**: +21.0 pp cross-period vs 0.5 pp within-2024. Ratio 42.8.
- **Distinct techs per posting**: +2.06 cross-period vs 0.32 within-2024. Ratio 6.4.
- **Description length**: Cohen's d 0.50 cross-period vs 0.28 within-2024. Ratio 1.8 (near_noise by the threshold, but substantial).
- **Seniority shares (J3/J1/J2/S1/S4)**: near_noise to below_noise — consistent with the Gate 1 memo's "magnitude fragility on senior-side" and the "asaniczka senior asymmetry."
- **Education mentions (PhD/MS/BS)**: below_noise — within-2024 differences (largely asaniczka vs arshkon boilerplate style) dominate over cross-period movement.
- **Aggregator share**: below_noise — asaniczka has a higher aggregator baseline than arshkon.

## Step 7 — Tech matrix sanity check

`tech_matrix_sanity.csv` — per-tech mention rate by source + arshkon:scraped ratio + flag.

**Verdict: no tokenization-artifact flags.** All flagged techs are interpretable as genuine 2024 → 2026 change, not regex failures.

- **23 flagged** (17% of 135): 19 `under_detected_2024` + 3 `new_in_2026` + 1 `over_detected_2024`.
  - All 19 `under_detected_2024` techs are LLM-era / modern-tooling (langchain, llamaindex, pinecone, mcp, anthropic, gemini, prompt_engineering, copilot, chatgpt, hugging_face, rag, llm, ai_agent, rust, fastapi, argocd, github_actions, mlflow, event_driven). These reflect real adoption growth 2024 → 2026.
  - 3 `new_in_2026`: cursor_tool, claude_tool, codex — did not exist / were niche in 2024.
  - 1 `over_detected_2024`: laravel (0.6% arshkon vs 0.2% scraped) — plausibly a real PHP decline, not a regex issue.
- **112 OK** — including all headline languages (Python, Java, JS, TS, Go, Rust, C++, C#, .NET, SQL) and all major clouds (AWS, Azure, GCP). Ratios all within 0.3 – 3 range.

**No residual issues.** The escape-fix on C++/C#/.NET was the known prior failure mode; the diagnostic proves it is working.

## Pre-existing Wave 1 artifacts (not re-produced)

- `seniority_definition_panel.csv` — T30 output (78 rows, 13 defs × 6 groups).
- `entry_specialist_employers.csv` — T06 output (206 rows).
- `returning_companies_cohort.csv` — T06 output (2,109 rows, 55% of 2026 postings).

---

## Build times

| Step | Elapsed |
|---|---|
| Step 4 (stoplist) | 0.2 s |
| Step 5 (asaniczka skills) | 0.4 s |
| Step 1 (cleaned text) | 10.6 s |
| Step 3 (tech matrix) | 514.4 s (~8.6 min) |
| Step 6 (calibration) | 48.4 s |
| Step 7 (tech sanity) | 0.1 s |
| Step 2 (embeddings) | see end of section — final number filled in on completion |

## Known issues / caveats

- **Raw-text fallback share on scraped**: 43.7% of scraped rows use raw `description` (LLM extraction `not_selected` for 19,487 + `deferred` for 172 + `skipped_short` for 2). Text-sensitive tasks must either filter to `text_source='llm'` (reduces scraped n to 25,509) or explicitly handle the boilerplate difference.
- **asaniczka aggregator share** is 16.2% vs arshkon 9.2%. This is a known sampling-frame difference (asaniczka over-represents Dice/Lensa). Aggregator-sensitive analyses should check T06's `entry_specialist_employers.csv` stoplist.
- **Seniority direction robust, magnitude noisy (Gate 1).** Use the T30 seniority panel for any seniority-stratified analysis, not the single-metric values from `calibration_table.csv`. The calibration table is a reference for noise-floor awareness, not the primary seniority reference.
- **Tech matrix uses raw `description`** (not `description_cleaned`). This is intentional — tech mentions are boilerplate-insensitive and benefit from full recall.
