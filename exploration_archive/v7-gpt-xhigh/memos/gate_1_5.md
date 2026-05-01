# Gate 1.5 Mechanical Check

## Artifact Status

Wave 1.5 shared preprocessing completed successfully after the Wave 1 OOM restart. The prep agent built the shared artifacts under `exploration/artifacts/shared/` and updated the report index.

Key artifacts present:

| Artifact | Status | Notes |
|---|---|---|
| `swe_cleaned_text.parquet` | Complete | 63,701 default-filtered LinkedIn SWE rows, 13 columns |
| `swe_embeddings.npy` | Complete | 34,258 x 384 float32 embeddings for `text_source = 'llm'` |
| `swe_embedding_index.parquet` | Complete | 34,258 rows matching embedding rows to `uid` |
| `swe_tech_matrix.parquet` | Complete | 63,701 rows, `uid` plus 148 technology/tool/practice indicators |
| `company_stoplist.txt` | Complete | 11,762 company-name tokens |
| `asaniczka_structured_skills.parquet` | Complete | 478,638 long-form skill rows from 18,114 asaniczka postings |
| `calibration_table.csv` | Complete | 48 metrics, 44 `ok`, 4 `ok_limited` |
| `tech_matrix_sanity.csv` | Complete | 148 technologies, 39 arshkon/scraped ratio flags |
| `README.md` | Complete | Includes row counts, coverage, sanity notes, and memory posture |

## Coverage Notes

The Gate 1 text-coverage constraint remains binding. Shared cleaned text uses LLM-cleaned text where available and raw fallback otherwise:

| Source | LLM text | Raw fallback |
|---|---:|---:|
| kaggle_arshkon | 4,687 | 4 |
| kaggle_asaniczka | 17,037 | 1,092 |
| scraped | 12,534 | 28,347 |

Wave 2 text-sensitive tasks must filter to `text_source = 'llm'` and report source-period coverage. Raw fallback is available only for boilerplate-insensitive binary recall checks.

## Memory Check

The prep run used the revised memory posture:

- DuckDB `PRAGMA memory_limit='4GB'`
- DuckDB `PRAGMA threads=1`
- Arrow row batches of 4,096
- Sentence-transformer batch size 64
- CPU-only embeddings with `CUDA_VISIBLE_DEVICES=''`
- Peak reported RSS about 1.47 GB

No Stage 9/10 reruns occurred and no new LLM calls were made.

## Tech Matrix Notes

Regex asserts covered known fragile patterns: `C++`, `C#`, `.NET`, `CI/CD`, Java vs JavaScript, and escaped markdown variants. Those known fragile patterns were not among the ratio flags. The 39 ratio-flagged technologies mostly look like real source/time composition candidates or terms requiring semantic validation rather than obvious tokenization failures.

Prevalence claims from the tech matrix still require semantic validation under the task-reference rules.

## Pass To Wave 2

Wave 2 agents should load shared artifacts rather than recomputing. Because of the confirmed OOM incident during Wave 1, Wave 2 should be staged or memory-capped rather than launching all heavy local modeling jobs at once. The highest-risk tasks are T09 and T15 because they use embeddings/topic modeling/dimensionality reduction.

