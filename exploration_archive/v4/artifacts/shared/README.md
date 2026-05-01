# Shared Analytical Artifacts

Built: 2026-04-09

These artifacts are pre-computed for Wave 2+ exploration agents to load directly, ensuring consistency and preventing duplicate computation.

## Data scope

All artifacts cover SWE LinkedIn rows with default filters (`source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true`).

- **Total rows:** 63,294
  - kaggle_arshkon: 5,019
  - kaggle_asaniczka: 23,213
  - scraped: 35,062

## Artifacts

### 1. `swe_cleaned_text.parquet` (77 MB)

Cleaned job description text with metadata columns.

**Columns:** `uid`, `description_cleaned`, `text_source`, `source`, `period`, `seniority_final`, `seniority_3level`, `is_aggregator`, `company_name_canonical`, `metro_area`, `yoe_extracted`, `swe_classification_tier`, `seniority_final_source`

**Text cleaning:** Company name stripping (exact match on `company_name_canonical`), URL/email removal, English stopword removal, whitespace normalization.

**STALE (frozen 2026-04-09, pipeline refactor 2026-04-10):** This artifact was built against the old pipeline that still produced a rule-based `description_core` column. At build time the fallback chain was `llm → rule → raw`, producing the distribution below. The rule-based stage was retired on 2026-04-10 and the builder now uses `llm → raw` only. Rebuild this artifact from the current pipeline before using it for any new text-sensitive analysis; boilerplate-sensitive consumers should filter to `text_source = 'llm'` regardless.

**WARNING (T12/T13, 2026-04-09):** The `description_cleaned` column has English stopwords removed, which destroys phrase-level analyses. Markdown headers like `**What you'll do**` become `** do**`; tone markers like `"you will"`, `"we are"` disappear entirely. For any analysis involving sentence/phrase structure (section classification, tone markers, readability, n-gram comparisons that depend on function words), use the raw text directly:

```sql
SELECT uid, COALESCE(description_core_llm, description) AS raw_text
FROM read_parquet('data/unified.parquet')
WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
```

The `description_cleaned` column is still fine for unigram bag-of-words tasks (keyword matching, tf-idf over content words, embedding inputs) where function words are not needed.

**`text_source` distribution (frozen 2026-04-09 build):**

| Source | llm | rule | raw | Total |
|---|---|---|---|---|
| kaggle_arshkon | 4,981 | 38 | 0 | 5,019 |
| kaggle_asaniczka | 20,831 | 2,382 | 0 | 23,213 |
| scraped | 7,269 | 27,793 | 0 | 35,062 |
| **Total** | **33,081 (52.3%)** | **30,213 (47.7%)** | **0** | **63,294** |

The "rule" column is historical. After a rebuild, `text_source` takes only `'llm'` or `'raw'` values — rows that were previously `'rule'` become `'raw'` (and should be treated accordingly by boilerplate-sensitive tasks).

Average cleaned text length: 2,683 characters.

### 2. `swe_embeddings.npy` (93 MB) + `swe_embedding_index.parquet` (711 KB)

Sentence-transformer embeddings using `all-MiniLM-L6-v2`.

- Shape: (63,294, 384), dtype: float32
- Normalized (L2 norm = 1.0)
- Input: first 2,560 chars of `description_cleaned` (model handles tokenization truncation)
- Build time: ~21 minutes on CPU (~52 rows/s)

**Usage:**
```python
import numpy as np
import pyarrow.parquet as pq

embeddings = np.load('exploration/artifacts/shared/swe_embeddings.npy')
index = pq.read_table('exploration/artifacts/shared/swe_embedding_index.parquet')
uids = index.column('uid').to_pylist()
# embeddings[i] corresponds to uids[i]
```

### 3. `swe_tech_matrix.parquet` (1.3 MB) — **KNOWN BUG, see below**

Binary matrix of 153 technology mentions detected via regex patterns in `description_cleaned`.

**KNOWN BUG (flagged by T12, 2026-04-09):** Undercounts C++ and C# (and any other token containing `+` or `#`). Verified rates:
- `c_cpp`: reports 0.5% in scraped (185 rows). Actual via DuckDB LIKE: 19.2% (~6,700 rows).
- `csharp`: reports 0.15% in scraped (54 rows). Actual: 11.3% (~3,977 rows).

Two root causes:
1. The regex `\bc\+\+\b` is fundamentally broken — `\b` doesn't match between `+` and end-of-word (non-word char to non-word char). The pattern only matches `c++X` where X is a word character, which is rare in practice.
2. Scraped LinkedIn text contains markdown backslash escapes (`C\+\+`, `C\#`, `.NET\-Core`) which need to be stripped before tokenization.

**Fix:** rebuild with (a) text un-escaped via `re.sub(r"\\([*+#.\-!()\[\]_`])", r"\1", text)`, (b) patterns using `(?:^|\W)c\+\+(?:\W|$)` or simple substring matching instead of `\b` boundaries.

**Impact:** Any Wave 1 claim depending on c_cpp, csharp, or other `+`/`#`-containing tokens needs recomputation. Other technologies (python, aws, javascript, etc.) are unaffected.

**Columns:** `uid` + 153 boolean technology columns.

**Technology categories:** Programming languages (28), frontend frameworks (14), backend frameworks (13), cloud/infrastructure (12), CI/CD & DevOps (11), databases & data (20), AI/ML frameworks (14), AI/LLM specific (13), AI tools (6), testing (10), methodologies (5), mobile (4), security (3).

**Top 10 technologies (prevalence):**

| Technology | Count | % |
|---|---|---|
| python | 27,511 | 43.5% |
| security | 23,887 | 37.7% |
| aws | 21,167 | 33.4% |
| java | 18,823 | 29.7% |
| agile | 18,609 | 29.4% |
| cicd | 17,507 | 27.7% |
| sql | 16,875 | 26.7% |
| javascript | 14,965 | 23.6% |
| azure | 14,614 | 23.1% |
| kubernetes | 12,605 | 19.9% |

### 4. `company_stoplist.txt` (80 KB)

10,652 tokens extracted from all unique `company_name_canonical` values, tokenized on whitespace/punctuation, lowercased, deduplicated. One token per line.

Used during text cleaning (Step 1) and available for downstream agents needing company-name-aware text processing.

### 5. `asaniczka_structured_skills.parquet` (5.6 MB)

Parsed skills from asaniczka SWE rows. Long format: one row per (uid, skill) pair.

- 609,894 skill mentions from 23,192 rows
- 122,533 unique skill strings
- Source: `skills_raw` column, comma-separated

### 6. `calibration_table.csv` (3.2 KB)

34 metrics comparing arshkon, asaniczka, and scraped. For each metric:

- Values per source
- Within-2024 effect size (arshkon vs asaniczka): Cohen's d (continuous) or Cohen's h (binary)
- Cross-period effect size (arshkon vs scraped)
- Calibration ratio: cross-period / within-2024

**Interpretation:** Calibration ratio > 1 means the cross-period difference exceeds the within-2024 baseline noise. Ratio near 1 means the cross-period difference is similar magnitude to within-2024 source variation (ambiguous). Negative ratios indicate the direction reversed.

**Key findings from calibration:**
- Description length: ratio 2.29 (real signal above within-2024 noise)
- Core length: ratio -33.41 (within-2024 near zero; cross-period large)
- Tech count: ratio 24.78 (negligible within-2024 variation, substantial cross-period increase)
- AI keyword prevalence: ratio -16.0 (direction reversed; cross-period shows strong increase)
- Python: ratio -157.26 (virtually no within-2024 diff; substantial cross-period increase)
- LLM mentions: ratio -7.42 (same pattern as AI keywords)

## Build scripts

- `build_shared_artifacts.py` — original monolithic script (Steps 0-6)
- `build_embeddings.py` — standalone embedding computation
- `build_remaining.py` — tech matrix, skills, calibration table

Total build time: ~25 minutes (embeddings dominate at ~21 min on CPU).
