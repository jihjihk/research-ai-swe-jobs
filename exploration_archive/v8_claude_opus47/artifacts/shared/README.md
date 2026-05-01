# Shared artifacts (Wave 1.5 / Agent Prep)

Built by the Wave 1.5 shared-preprocessing agent on 2026-04-17. All files under
this directory are Wave 2+ inputs: load them directly instead of recomputing.

Source parquet: `data/unified.parquet` (6.6 GB). Default filter applied to all
Wave-1.5 row-level artifacts:

```sql
WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
```

Total SWE LinkedIn rows after filter: **63,701**
(arshkon=4,691; asaniczka=18,129; scraped 2026-03=19,777; scraped 2026-04=21,104)

---

## File inventory

| File | Build step | Size | Rows | Cols |
|---|---|---|---|---|
| `company_stoplist.txt` | 1 | 477 KB | 61,868 tokens | 1 (raw) |
| `swe_cleaned_text.parquet` | 2 | 65.0 MB | 63,701 | 13 |
| `swe_embeddings.npy` | 3 | 50.0 MB | 34,102 × 384 | — |
| `swe_embedding_index.parquet` | 3 | 676 KB | 34,102 | 2 (row_idx, uid) |
| `swe_tech_matrix.parquet` | 4 | 1.13 MB | 63,701 | 108 (1 uid + 107 techs) |
| `tech_matrix_sanity.csv` | 4 | 11 KB | 107 | 7 |
| `asaniczka_structured_skills.parquet` | 5 | 4.3 MB | 475,504 (long form) | 2 (uid, skill) |
| `asaniczka_skill_frequency.csv` | 5 | 4.7 MB | 98,334 | 3 |
| `calibration_table.csv` | 6 | 4.6 KB | 21 metrics | 14 |
| `seniority_definition_panel.csv` | T30 (preexisting) | 11 KB | — | — |
| `entry_specialist_employers.csv` | T06 (preexisting) | 60 KB | — | — |

**Total directory size: 127 MB**

---

## 1. `company_stoplist.txt`

One token per line, lowercased, deduplicated. Extracted from all unique
`company_name_canonical` values (117,726 distinct names) by splitting on
whitespace + punctuation (`,.-_&()[]/'"|`). Wave 2 tokenizers use this as the
reference set for company-name stripping.

Note: the stoplist is written verbatim — it contains common English words
(`the`, `and`, `of`, ...) and common SWE-JD vocabulary (`ai`, `data`,
`design`, `engineer`, ...) that appear inside company names. **Consumers
must apply their own English-stopword and tech-vocabulary guards before
using this file as a stripping set** — otherwise they will strip
substantial semantic content from descriptions. The Wave 1.5 cleaned-text
artifact already applies this guard internally (see step 2).

---

## 2. `swe_cleaned_text.parquet`

Row-per-posting cleaned text for every default-filtered SWE LinkedIn row.

Columns:

| Column | Type | Notes |
|---|---|---|
| `uid` | string | Primary key |
| `description_cleaned` | string | `description_core_llm` if LLM-labeled, else raw `description`; lowercased, whitespace-collapsed, markdown-escapes unescaped, company-name tokens stripped |
| `text_source` | string | `'llm'` or `'raw'` — Wave 2 text-sensitive analyses should filter to `'llm'` |
| `source` | string | `kaggle_arshkon`, `kaggle_asaniczka`, `scraped` |
| `period` | string | `2024-04`, `2024-01`, `2026-03`, `2026-04` |
| `seniority_final` | string | 5-level enum |
| `seniority_3level` | string | junior/mid/senior/unknown |
| `seniority_final_source` | string | `title_keyword`, `title_manager`, `llm`, `unknown` |
| `is_aggregator` | bool | — |
| `company_name_canonical` | string | — |
| `metro_area` | string | — |
| `yoe_extracted` | double | — |
| `swe_classification_tier` | string | `regex`, `embedding_high`, `title_lookup_llm`, `embedding_adjacent` |

### Markdown unescape fix

Regex: `re.sub(r"\\([+\-#.&_()\[\]\{\}!*])", r"\1", text)`

Applied unconditionally before lowercasing and tokenization. Sanity asserts
pass:

```python
_unescape(r"C\+\+") == "C++"
_unescape(r"C\#") == "C#"
_unescape(r"\.NET") == ".NET"
```

Rows affected (raw description with `\+`, `\#`, or `\.`):

| Source | raw rows with `\+\+` | raw rows with `c\#` | raw rows with `\.net` |
|---|---|---|---|
| kaggle_arshkon | 0 | 0 | 0 |
| kaggle_asaniczka | 0 | 0 | 0 |
| scraped | 7,737 | 0 | 0 |

Only scraped text has markdown escapes. Post-fix, `description_cleaned` has
zero backslash-escape residuals on arshkon/asaniczka and 17/40,881 residuals
on scraped (unusual markdown combinations we did not cover — negligible).

### text_source split per (source × period)

| source | period | llm | raw | % llm |
|---|---|---|---|---|
| kaggle_arshkon | 2024-04 | 4,653 | 38 | 99.2% |
| kaggle_asaniczka | 2024-01 | 16,993 | 1,136 | 93.7% |
| scraped | 2026-03 | 6,282 | 13,495 | 31.8% |
| scraped | 2026-04 | 6,174 | 14,930 | 29.3% |

Scraped LLM coverage is much lower than Kaggle because the scraped dataset
keeps growing and not every row has been routed through Stage 9 yet.
**Wave 2 text-sensitive analyses MUST filter to `text_source = 'llm'`** or
they will compare LLM-cleaned 2024 text against raw 2026 text and the
length / boilerplate differences will dwarf the signal.

### Company-name stripping policy

The cleaned-text build uses two passes:

1. **Global stopset** — tokens from `company_stoplist.txt` of length ≥ 3 that
   are NOT on a 483-token preserve list (English function words, core tech
   taxonomy, generic SWE-JD vocabulary like "design", "build", "manage",
   "engineer", "team", etc.). Any whitespace-separated token in the cleaned
   text whose punctuation-stripped form is in the global stopset is dropped.
2. **Per-row stopset** — the posting's own `company_name_canonical` is
   tokenized and joined with the global stopset for this row only, so
   one-off brand names that are too rare to appear in the global set are
   still stripped.

Preserved: punctuation inside tokens (`c++`, `c#`, `.net`) and common English
stopwords (`the`, `and`, ...). Wave 2 tasks can apply their own stopword
removal on top.

---

## 3. `swe_embeddings.npy` + `swe_embedding_index.parquet`

MiniLM `all-MiniLM-L6-v2` 384-dim embeddings on `description_cleaned` for
rows where `text_source='llm'` only. Truncated at 2,500 chars (~500 tokens)
before encoding. Batched at 256.

- `swe_embeddings.npy`: float32, shape `(N, 384)` where N = LLM-labeled row count.
- `swe_embedding_index.parquet`: `row_idx` (int64, 0..N-1) + `uid`. Join on
  `uid` to pull source/period/seniority_final/etc from the cleaned-text
  artifact.

---

## 4. `swe_tech_matrix.parquet` + `tech_matrix_sanity.csv`

### Matrix

One row per SWE LinkedIn `uid`. Columns: `uid` + 107 snake_cased boolean
tech flags. True if ANY regex variant for that tech fires on
`description_cleaned`.

Taxonomy coverage (107 techs):

- Languages (18): python, java, javascript, typescript, go, rust, cpp,
  csharp, ruby, kotlin, swift, scala, php, r_language, perl, bash, shell, sql
- Frontend (10): react, angular, vue, nextjs, svelte, jquery, html, css,
  redux, tailwind
- Backend (9): nodejs, django, flask, spring, dotnet, rails, fastapi,
  express, laravel
- Cloud / DevOps (13): aws, azure, gcp, kubernetes, docker, terraform,
  ansible, cicd, jenkins, github_actions, argocd, gitlab, helm
- Data (15): postgresql, mysql, mongodb, redis, kafka, spark, snowflake,
  databricks, dbt, elasticsearch, oracle, dynamodb, cassandra, bigquery
- ML/AI traditional (8): tensorflow, pytorch, sklearn, pandas, numpy,
  jupyter, keras, xgboost
- ML/AI LLM-era (14): langchain, langgraph, rag, vector_database, pinecone,
  chromadb, huggingface, openai, claude, gemini, mcp, llamaindex,
  anthropic, ollama
- AI tools (8): copilot, cursor, chatgpt, codex, llm_token,
  prompt_engineering, fine_tuning, agent_framework
- Testing (7): jest, pytest, selenium, cypress, junit, mocha, playwright
- Practices (6): agile, scrum, tdd, devops, sre, microservices

### Sanity verdict

26 of 107 techs flagged on arshkon→scraped ratio ∉ [0.33, 3]:

- **21 AI/LLM-era tools** (`langchain`, `rag`, `pinecone`, `claude`,
  `openai`, `copilot`, `cursor`, etc.): expected real 2024→2026 market
  shift, NOT a tokenization defect. These are the headline finding the
  calibration table stress-tests.
- **4 rapid-adoption frameworks** (`rust`, `fastapi`, `github_actions`,
  `argocd`): real adoption growth between the 2024 Kaggle snapshots and
  the 2026 scrape.
- **1 declining tech** (`laravel`, ratio 3.7×): likely real PHP/Laravel
  share decline; could also be sampling noise in scraped.

**Critical: none of the escape-sensitive techs (`cpp`, `csharp`,
`dotnet`) are flagged.** Their arshkon/scraped ratios are 0.58, 1.10,
and 1.62 — all within the 0.33–3 band — confirming the markdown unescape
fix worked. See `tech_matrix_sanity.csv` for per-tech rates and notes.

---

## 5. `asaniczka_structured_skills.parquet` + `asaniczka_skill_frequency.csv`

Parses asaniczka SWE rows' `skills_raw` field (comma-separated) into:

- **Long form** `uid | skill` (475,504 rows) — one row per uid × skill pair.
  Lowercased, trimmed, deduped within each posting.
- **Frequency CSV** with columns `skill | n_postings | share_of_asaniczka_swe`.
  98,334 unique skill strings observed.

Top skills: `python` (31%), `java` (26%), `sql` (20%), `aws` (19%),
`javascript` (17%), `software engineering` (16%), `software development`
(15%), `communication` (14%), `git` (14%), `agile` (13%), `linux` (13%),
`kubernetes` (12%), `docker` (12%), `c++` (11%), `c#` (10%).

Note: asaniczka structured skills have `c++` (11%) and `c#` (10%) as
free-text labels but asaniczka *description* text rarely mentions
`c#` (0.01%) — the source appears to put tech mentions in the structured
skills field rather than narrative text. T14 will compare description-
extracted tech frequencies against these structured baselines.

---

## 6. `calibration_table.csv` (keystone)

For each of 21 metrics, computes arshkon/asaniczka/scraped values, within-2024
effect (arshkon − asaniczka), pooled 2024 value, cross-period effects
(arshkon→scraped and pooled→scraped), and SNR verdict. Columns named per
spec so Wave 2 agents can `pd.read_csv` without guessing.

Verdict: `above_noise` if max(SNR) ≥ 2; `below_noise` if < 1; `marginal`
otherwise.

### Above-noise metrics (13)

| metric | arshkon | asaniczka | scraped | within | cross(pooled) | SNR | verdict |
|---|---|---|---|---|---|---|---|
| `description_length_mean` | 3306 | 3881 | 4912 | -576 | +1149 | 2.0 | above |
| `description_length_median` | 2974 | 3733 | 4862 | -759 | +1285 | 1.7 | above |
| `description_cleaned_length_mean` | 1329 | 1559 | 2630 | -230 | +1119 | 4.9 | above |
| `description_cleaned_length_median` | 1237 | 1421 | 2422 | -184 | +1039 | 5.6 | above |
| `tech_count_mean` | 5.38 | 5.10 | 6.95 | +0.28 | +1.80 | 6.3 | above |
| `ai_mention_binary_share` | 16.4% | 14.9% | 51.3% | +1.5pp | +36.1pp | **24.7** | above |
| `ai_mention_density_per_1k` | 0.61 | 0.38 | 1.55 | +0.22 | +1.12 | 5.0 | above |
| `ai_tool_binary_share` | 1.5% | 1.1% | 14.9% | +0.4pp | +13.7pp | **35.4** | above |
| `management_strict_binary_share` | 34.4% | 40.4% | 48.2% | -6.0pp | +9.0pp | 1.5 | above |
| `org_scope_binary_share` | 39.8% | 41.5% | 62.2% | -1.7pp | +21.1pp | 12.7 | above |
| `soft_skill_binary_share` | 59.0% | 60.4% | 68.6% | -1.3pp | +8.5pp | 6.5 | above |

### Marginal / below-noise metrics (8)

- `tech_count_median` (SNR 1.8): marginal — medians are close.
- `tech_count_density_per_1k` (SNR 1.0): marginal; scraped is LOWER per 1K
  chars because descriptions are much longer (length dominates).
- `management_broad_binary_share` (SNR 0.48): below-noise; mgmt language
  very prevalent in 2024 baseline already.
- `aggregator_share` (SNR 0.26): below-noise; arshkon has unusually low
  aggregator share vs asaniczka.
- `yoe_known_share` (SNR 0.18): below-noise.
- `j1_entry_share` (SNR 0.56): below-noise — within-2024 effect (arshkon's
  native entry labels) dominates the cross-period shift.
- `j2_entry_or_associate_share` (SNR 0.51): below-noise for same reason.
- `j3_yoe_leq_2_share` (SNR 1.3): marginal.
- `s1_senior_share` (SNR 0.84): below-noise; asaniczka has a very high
  senior share by construction.
- `yoe_extracted_median`: undefined (all three sources at median=5 years).

### Take-home

The **AI-mention and AI-tool rise from 2024 to 2026 is massively
above-noise** — SNRs of 24.7 and 35.4 on the within-2024 calibrator. This
corroborates Gate 1's keystone +33pp AI-mention finding across a broader
metric set. Seniority-share metrics (J1/J2) are below-noise because the
within-2024 noise is large (arshkon's unique native-label coverage inflates
its entry rate relative to asaniczka). Wave 2 agents J-M should rely on
the T30 seniority panel and the YOE-based proxy for seniority-dependent
claims, and on this calibration table for text-mining claims.

---

## Preexisting artifacts (cross-reference)

- `seniority_definition_panel.csv` — T30 seniority definition sensitivity
  panel. Rows per seniority definition (J1..J5, S1..S3) × period source,
  with MDE and direction-of-effect columns. See T30 memo in
  `exploration/reports/`.
- `entry_specialist_employers.csv` — T06 decomposition output: 125-company
  arshkon∩scraped overlap panel with per-company within-company shift on
  entry share and AI-mention share. See T06 memo in
  `exploration/reports/`.

---

## Known limits / caveats

- **Scraped LLM coverage is ~30%.** Tasks requiring LLM-cleaned text must
  restrict to `text_source = 'llm'` (see column in the cleaned-text
  artifact). A Wave 2 ablation that includes the `raw` subset will over-
  count boilerplate and inflate length-based metrics.
- **Asaniczka has zero native entry labels.** Any seniority-stratified
  analysis pooling asaniczka depends entirely on `seniority_final` (LLM
  path) for entry-level rows. The YOE-based proxy is the only fully label-
  independent validator — use it as the primary check.
- **The AI-mention regex in the calibration table is broad.** The `ai` and
  `agent` tokens will trigger on many non-LLM contexts (e.g. "AI-assisted",
  "agent-based simulation"). This is the right baseline for calibration.
  Wave 2's T22 will refine with validated patterns.
- **Backslash-escape residuals on scraped:** 17 rows out of 40,881 still
  contain `\[+#.]` in the cleaned text after unescape. These are rare
  markdown patterns we did not special-case (e.g. `\{foo\}`). Negligible
  for the downstream tech regex.
