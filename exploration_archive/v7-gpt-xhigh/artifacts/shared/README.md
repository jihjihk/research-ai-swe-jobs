# Wave 1.5 Shared Preprocessing Artifacts

Built: `2026-04-16T02:59:39.258279+00:00`

## Inventory

| Artifact | Rows / columns | Notes |
|---|---:|---|
| `swe_cleaned_text.parquet` | 63,701 rows / 13 columns | Default LinkedIn, English, date-ok, SWE rows. Uses LLM cleaned text when labeled, otherwise raw fallback. |
| `swe_embeddings.npy` | 34,258 rows x 384 dims | all-MiniLM-L6-v2 embeddings for `text_source = 'llm'`, first 512 whitespace tokens, float32. |
| `swe_embedding_index.parquet` | 34,258 rows | Row index to `uid` mapping for embeddings. |
| `swe_tech_matrix.parquet` | 63,701 rows / 149 columns | `uid` plus boolean technology/tool/practice indicators. |
| `company_stoplist.txt` | 11,762 tokens | Lowercased tokens from `company_name_canonical` in the default SWE LinkedIn frame. |
| `asaniczka_structured_skills.parquet` | 478,638 rows | Long-form parsed skills from `skills_raw`. |
| `calibration_table.csv` | 48 metrics | Within-2024 and arshkon-to-scraped lightweight calibration metrics. |
| `tech_matrix_sanity.csv` | 148 technologies | Source-specific mention rates and arshkon/scraped ratio flags. |
| `tech_taxonomy.csv` | 148 rows | Regex definitions used for the tech matrix. |
| `prep_build_metadata.json` | 1 JSON document | Build parameters, counts, and memory posture. |

## Row Counts And Text Sources

| source           | text_source | rows  |
| ---------------- | ----------- | ----- |
| kaggle_arshkon   | llm         | 4687  |
| kaggle_arshkon   | raw         | 4     |
| kaggle_asaniczka | llm         | 17037 |
| kaggle_asaniczka | raw         | 1092  |
| scraped          | llm         | 12534 |
| scraped          | raw         | 28347 |

The scraped LinkedIn cleaned-text constraint from Gate 1 remains binding: roughly two-thirds of scraped SWE rows use raw text fallback in this shared artifact. Text-sensitive downstream tasks should filter to `text_source = 'llm'` and report coverage.
97 rows had `llm_extraction_coverage = 'labeled'` but null `description_core_llm`; per the task definition they remain `text_source = 'llm'` with empty cleaned text. Total empty `description_cleaned` rows: 157.

## Embedding Coverage

- Target rows (`text_source = 'llm'`): 34,258
- Rows embedded: 34,258
- Complete: `True`
- Batch size: 64
- `CUDA_VISIBLE_DEVICES`: ``

## Tech Sanity

- Technology columns: 148
- Ratio-flagged technologies: 39

| technology    | label         | arshkon_to_scraped_ratio | investigation_note                                                                                            |
| ------------- | ------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| agents        | Agents        | 0.0086370396129991       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| anthropic_api | Anthropic API | 0.0                      | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| argo_cd       | Argo CD       | 0.2089873613792828       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| buildkite     | Buildkite     | 0.2954160328649524       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| c_language    | C             | 40.08795565977403        | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| chatgpt       | ChatGPT       | 0.3227693692413368       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| chroma        | Chroma        | 0.0                      | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| claude        | Claude        | 0.0061242255583387       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| claude_api    | Claude API    | 0.0                      | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| codex         | Codex         | 0.0                      | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| copilot       | Copilot       | 0.0                      | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| cursor        | Cursor        | 0.0                      | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| dart          | Dart          | 0.2857302612956096       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| elixir        | Elixir        | 0.2954160328649524       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| evals         | Evals         | 0.022753976421713        | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| fastapi       | FastAPI       | 0.1801748492065351       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| gemini        | Gemini        | 0.0197613899535512       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| generative_ai | Generative AI | 0.1753134082121991       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| hugging_face  | Hugging Face  | 0.0523934246664294       | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |
| langchain     | LangChain     | 0.0                      | flagged; not a known markdown-escape edge case, likely composition/source difference or needs semantic review |

Regex edge-case asserts ran before scanning and covered escaped markdown variants of `C++`, `C#`, `.NET`, `CI/CD`, plus Java vs JavaScript boundaries. Residual ratio flags are documented in `tech_matrix_sanity.csv`; most are expected to reflect real source/time composition or require semantic review rather than escaped-token undercounting.

## Calibration Table

`calibration_table.csv` contains 48 metrics. Rows marked `ok_limited` are computable but use lightweight regex indicators or have undefined calibration ratios because within-2024 variation is zero.

## Memory-Safety Notes

- DuckDB connections set `PRAGMA memory_limit='4GB'` and `PRAGMA threads=1`.
- The script selects only the default LinkedIn SWE columns needed for artifacts.
- Arrow row batches: 4,096.
- Sentence-transformer batches: 64, CPU-only.
- `data/unified.parquet` was not loaded wholesale into pandas.
- Company-token and English-stopword stripping preserves protected technology tokens such as `Go`, `C++`, `.NET`, `OpenAI`, `R`, and `CI/CD` so the requested tech indicators are not erased before scanning.
- Peak process RSS reported by `resource.getrusage`: 1474.5 MB.

## Known Limitations From Gate 1

- Scraped LinkedIn LLM cleaned-text coverage is low; raw fallback is present by design but is not valid for boilerplate-sensitive claims.
- `seniority_final` is conservative and the unknown seniority pool is large and structured.
- Asaniczka native `associate` is not a junior proxy; do not use native asaniczka seniority as an entry baseline.
- Company composition is a first-order confound; corpus-level downstream tasks need company caps and aggregator sensitivity.
- The tech matrix is regex-based and intended as a shared binary mention screen. Prevalence claims still need semantic validation under the task-reference rules.
