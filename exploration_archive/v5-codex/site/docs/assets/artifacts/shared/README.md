# Shared preprocessing artifacts

Build date: 2026-04-11 15:58:34 PDT
Build time: about 12 minutes

## Contents

| Artifact | Path | Rows / shape | Notes |
|---|---|---:|---|
| Cleaned text | `swe_cleaned_text.parquet` | 59,972 rows | `description_cleaned`, `text_source`, and wave-friendly metadata for SWE LinkedIn rows |
| Embeddings | `swe_embeddings.npy` | 26,219 rows × 384 dims | complete |
| Embedding index | `swe_embedding_index.parquet` | 26,219 rows | Maps embedding row index to `uid` |
| Technology matrix | `swe_tech_matrix.parquet` | 59,972 rows × 182 cols | Binary mention matrix from cleaned text |
| Company stoplist | `company_stoplist.txt` | 60,566 tokens | One token per line |
| Structured skills | `asaniczka_structured_skills.parquet` | 475,525 rows | Parsed comma-separated `skills_raw` for asaniczka SWE |
| Calibration table | `calibration_table.csv` | 43 rows | Within-2024 vs arshkon-vs-scraped calibration metrics |

## SWE LinkedIn coverage

Total rows: 59,972

Text source distribution:
text_source
raw    33753
llm    26219

By source:
          source text_source     n
  kaggle_arshkon         llm  4679
  kaggle_arshkon         raw    12
kaggle_asaniczka         llm 15574
kaggle_asaniczka         raw  2555
         scraped         llm  5966
         scraped         raw 31186

LLM extraction coverage by source:
          source  labeled   raw  labeled_share
  kaggle_arshkon     4679    12       0.997442
kaggle_asaniczka    15574  2555       0.859066
         scraped     5966 31186       0.160584

Distinct periods by source:
          source  period     n
  kaggle_arshkon 2024-04  4691
kaggle_asaniczka 2024-01 18129
         scraped 2026-03 19796
         scraped 2026-04 17356

## Notes

- `text_source = 'llm'` uses `description_core_llm`; `text_source = 'raw'` falls back to raw `description` after company-name and stopword stripping.
- Company stopwords are drawn from all `company_name_canonical` tokens across the unified dataset. A small protected token set preserves obvious technology tokens such as `go`, `r`, `c`, `.net`, `node.js`, and `next.js`.
- Embeddings are computed only for `text_source = 'llm'` rows, truncated to the first 512 tokens before encoding with `all-MiniLM-L6-v2`.
- The calibration table uses all SWE LinkedIn rows under the default filters and compares arshkon to asaniczka within 2024, then arshkon to scraped for the current cross-period signal.
- Rows with `llm_extraction_coverage = 'labeled'` are the only rows with LLM-cleaned text. Thin coverage remains a limitation for text-heavy downstream tasks.
- Embedding coverage: 43.7% of cleaned SWE LinkedIn rows.
