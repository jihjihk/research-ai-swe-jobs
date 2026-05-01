# Shared analytical artifacts (Wave 1.5)

Built: 2026-04-15 by Agent Prep
Input: `data/unified.parquet` (preprocessing V4 / pipeline run through Stage 10)
Default filter applied throughout: `is_swe = true AND source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'`

These artifacts replace per-task recomputation for Wave 2+ agents. Load them — do not recompute.

---

## Contents

| File | Row count | Notes |
|---|---|---|
| `swe_cleaned_text.parquet` | 63,701 | All SWE LinkedIn rows passing default filter. Has `uid`, `description_cleaned`, `text_source`, `source`, `period`, `seniority_final`, `seniority_3level`, `is_aggregator`, `company_name_canonical`, `metro_area`, `yoe_extracted`, `swe_classification_tier`, `seniority_final_source`. |
| `swe_embeddings.npy` | 34,099 | `float32`, shape `(34099, 384)`. `all-MiniLM-L6-v2`, normalized. **Only `text_source='llm'` rows** (step-2 spec). |
| `swe_embedding_index.parquet` | 34,099 | `row_idx → uid` mapping, aligns with `swe_embeddings.npy`. |
| `swe_tech_matrix.parquet` | 63,701 | `uid` + 123 boolean technology columns. Covers ALL SWE rows (LLM + raw text) so binary-keyword downstream tasks are not text-coverage-capped. |
| `asaniczka_structured_skills.parquet` | 478,638 (long) | Per-skill rows parsed from `skills_raw` for 18,114 distinct asaniczka SWE uids. |
| `calibration_table.csv` | 45 metrics | Within-2024 (arshkon vs asaniczka) and cross-period (arshkon vs scraped) effect sizes and SNR ratio. |
| `company_stoplist.txt` | 61,246 tokens | Lowercased tokens from `company_name_canonical` across ALL rows (not just SWE). A ~65-item tech-term whitelist overrides this stoplist during cleaning so tokens like `python`, `aws`, `react` are never stripped. |

---

## `text_source` distribution by source × period

| source | period | llm | raw | llm share |
|---|---|---|---|---|
| kaggle_arshkon  | 2024-04 | 4,687  | 4      | 99.9% |
| kaggle_asaniczka | 2024-01 | 17,037 | 1,092  | 94.0% |
| scraped         | 2026-03 | 6,325  | 13,452 | 32.0% |
| scraped         | 2026-04 | 6,209  | 14,895 | 29.4% |
| **scraped total** | **2026** | **12,534** | **28,347** | **30.7%** |

**Binding constraint (confirmed):** Scraped 2026 text-labeled n = **12,534** (30.7% of 40,881 SWE scraped rows). Every text-based analysis that needs cleaned (boilerplate-stripped) text is capped at this count on the 2026 side. Per Gate 1 decision 4, raw-text backfill is only acceptable for binary keyword presence.

---

## Which downstream tasks filter to `text_source='llm'` vs use all rows

**Filter to `text_source='llm'` (boilerplate-sensitive):**
- T08 length distributions, topic models, corpus comparisons
- T09 archetype discovery (BERTopic, NMF)
- T10 title taxonomy evolution (when using description content)
- T11 requirements complexity
- T12 corpus comparison
- T13 section anatomy, readability, tone
- T14 technology ecosystem TF-IDF / co-occurrence (but see note below)
- T15 semantic similarity landscape (uses shared embeddings)

**Use all rows (binary keyword presence — boilerplate-insensitive):**
- T14 technology presence vectors (share any mention) — `swe_tech_matrix.parquet` covers all 63,701 rows
- Any "does this posting mention X anywhere" metric
- Non-text metrics (seniority counts, YOE, geography, company analysis)

**Never mix without reporting the split.** Per the analytical preamble's text source discipline, any analysis that compares two subsets where one used llm-cleaned text and the other used raw text must report the split and test sensitivity.

---

## Build notes

- **Company stoplist.** Built from 117,726 distinct `company_name_canonical` values tokenized on whitespace + common punctuation (`,./-_&'()|+:;!?"*<>`). Tokens < 2 chars or pure-digit are dropped. A 65-item tech-term whitelist (`python`, `java`, `aws`, `react`, `docker`, `llm`, `openai`, `anthropic`, `cursor`, `copilot`, `claude`, `gemini`, `mcp`, ...) overrides the stoplist. Without this whitelist, ~30 tech tokens would be stripped from descriptions because they appear in company names (e.g., "Python Software Foundation", "OpenAI", "Cursor Inc.").

- **Cleaned text tokenizer.** Regex `[A-Za-z][A-Za-z0-9+#./-]*` preserves `C++`, `C#`, `Node.js`, `.NET`, `CI/CD` as single tokens. All text is lowercased before matching. Standard English stopwords (NLTK, 198 tokens) are then removed. A few rows have empty cleaned text after stripping (very short raw descriptions); these are excluded from the embedding step.

- **Embeddings.** `sentence-transformers/all-MiniLM-L6-v2` with `normalize_embeddings=True` and model `max_seq_length = 256` (hard cap for this model; the spec says "first 512 tokens" but the model truncates to 256 wordpieces internally — text was pre-truncated to 512 whitespace tokens). Batch size 256. Ran on CPU. ~11 minutes for 34,099 rows. No OOM. 8 rows with empty cleaned text were dropped (present as `text_source='llm'` in the cleaned-text parquet but excluded from embeddings).

- **Tech matrix.** 123 regex patterns scanning the padded cleaned text (` ` + text + ` `). Patterns use `(?:^|\s)TECH(?:$|\s)` form instead of `\b` because `\b` is unreliable near `+`, `#`, `.`, `/`. Inline assertions verify `c++`, `c#`, `.NET`, `node.js`, `ci/cd`, `k8s`, `golang`, `java`-vs-`javascript` exclusion, and `scala`-vs-`scalable` exclusion all behave correctly. **All tech pattern assertions passed.** See `exploration/scripts/prep_04_tech_matrix.py` for the full taxonomy.

- **Calibration table.** 45 metrics across description length, YOE, tech counts, AI prevalence (3 variants), management (broad + strict), org-scope, soft-skill, credential, education, seniority entry share (of known and of all), text-labeled share, and 28 specific tech prevalences. Continuous metrics use pooled-variance Cohen's d; binary metrics use absolute proportion difference. `calibration_ratio = cross_period_effect / within_2024_effect`; ratio > 2 is above instrument noise per the T05 SNR convention. Ratio-`inf` rows are metrics where the within-2024 effect is numerically zero — treat as strong-signal but note the instability.

---

## Top 10 most-prevalent technologies (full SWE LinkedIn corpus, n=63,701)

| rank | technology | count | share |
|---|---|---|---|
| 1 | python      | 25,872 | 40.6% |
| 2 | aws         | 18,903 | 29.7% |
| 3 | java        | 17,530 | 27.5% |
| 4 | agile       | 16,960 | 26.6% |
| 5 | cicd        | 16,280 | 25.6% |
| 6 | sql_lang    | 15,163 | 23.8% |
| 7 | azure       | 13,112 | 20.6% |
| 8 | rest_api    | 11,203 | 17.6% |
| 9 | kubernetes  | 10,578 | 16.6% |
| 10 | javascript | 9,584  | 15.0% |

---

## Calibration table highlights

**Top 5 strongest signal-to-noise (calibration_ratio ignoring inf/unstable):**
- `management_indicator_rate_broad`: 57.5 (but see keyword-validation warning below)
- `tech_nodejs_prevalence`: 56.8
- `tech_copilot_prevalence`: 43.5
- `tech_postgres_prevalence`: 34.6
- `tech_gcp_prevalence`: 33.1

**Bottom 5 noisiest (below-threshold, not cleanly cross-period):**
- `seniority_final_entry_share_of_known`: 0.17 — confirms Gate 1 finding that entry-share cross-period is below instrument noise
- `seniority_final_entry_share_of_all`: 0.23
- `yoe_extracted_mean` / `yoe_extracted_median`: 0.34
- `yoe_le2_share`: 0.48
- `soft_skill_rate`: 0.02 (cross-period effect is ~0 — no meaningful cross-period change)

**Important caveat on management_indicator_rate_broad (SNR 57.5):** The broad management pattern includes `\blead\b`, `\bleading\b`, etc. Prior waves found these generic patterns inflate management indicators by 3-5× because they match "a leading company" instead of "leading a team." Agent L (T21) has been instructed to validate this pattern on a 50-row sample. Treat the broad SNR as upper-bound until validated. The `management_indicator_rate_strict` SNR (8.4) is a more defensible lower-bound.

**Consistency with T05 SNR table:**
- description_length SNR: this table 2.51 vs T05 2.29 (close — small diff from Cohen's d vs raw abs diff). Marginal.
- entry_share SNR: this table 0.17 vs T05 0.33. Both below 2 — below-threshold consensus stands.
- yoe_le2 SNR: this table 0.48 vs T05 0.98. Both below 2.
- ai_mention SNR: T05 reported 925 using a narrower AI keyword set; this table's broader `ai_keyword_prevalence_any` gives 18.8, `ai_tool_specific_prevalence` gives 31.6. Both are well above 2; AI signal remains the single strongest cross-period signal.

---

## Coverage gaps and surprises

- **Embedding row count (34,099)** is slightly higher than the sum of labeled rows in the cleaned-text parquet (34,258 minus ~160 rows with empty cleaned text post-stopword-stripping). The 160-row gap is dominated by very short arshkon descriptions that reduce to a single word or nothing after removing English stopwords + company tokens. This is acceptable — those rows are too short for embeddings to be meaningful anyway.
- **~1,092 asaniczka rows are `text_source='raw'`.** These are rows where Stage 9 LLM extraction was not applied (router-excluded or budget-capped). They're included in the tech matrix but not the embeddings.
- **No OOM on the embedding step.** The batched computation stayed well under the 31 GB limit (peak RSS observed ~2 GB for this workload).
- **Early build had a tech-in-stoplist bug.** First pass stripped ~30 tech tokens (`python`, `aws`, `java`, ...) because company names contain them. Fixed by the `TECH_PROTECT` whitelist in `prep_02_cleaned_text.py`. Confirmed tech counts now match expectations (Python 40.6%, AWS 29.7%).

---

## Recommendations for Wave 2+ agents

1. **Do NOT recompute embeddings.** The shared file is complete and covers every `text_source='llm'` row. Load `swe_embeddings.npy` + `swe_embedding_index.parquet` directly.
2. **Do NOT rebuild the tech matrix.** 123 columns, tested regex patterns, covers all 63,701 SWE rows. Use `tech_count_mean` style derivations by summing columns per row.
3. **Do NOT recompute within-2024 calibration for standard metrics.** Load `calibration_table.csv` and cite the ratio. Only compute extra metrics that are not already in the table.
4. **Filter to `text_source='llm'`** for topic models, term frequencies, length distributions, and anything corpus-level. This is the 34k-row cleaned-text frame.
5. **The `seniority_final` entry-share cross-period comparison is below instrument noise (SNR 0.17).** Any entry-level claim must report `seniority_final` + YOE ≤ 2 side by side, per Gate 1 decision 2.
6. **Cap 50 postings per `company_name_canonical`** for corpus-level term-frequency and topic-model work per T06's concentration prediction table.
7. **The broad management indicator needs sample validation.** SNR 57.5 looks strong but has ~3-5× inflation risk; Agent L (T21) should validate on a 50-row sample before any management finding is reported.

---

## Build time

- Stoplist: 0.1 s
- Cleaned text: 10.2 s
- Tech matrix: 85 s
- Structured skills: 0.7 s
- Calibration: 7.1 s
- Embeddings: 683 s (~11 min, CPU, single-threaded)
- **Total:** ~13 minutes

Scripts: `exploration/scripts/prep_0[1-6]_*.py`. Rerun any step by invoking its script with `./.venv/bin/python`. Steps are independent except: embeddings + tech matrix + calibration all depend on `swe_cleaned_text.parquet`, which depends on `company_stoplist.txt`.
