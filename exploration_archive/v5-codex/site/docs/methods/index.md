# Methods and data.

This section explains how the heterogeneous inputs became a comparable corpus, how to read the resulting columns, and how to interpret the findings without over-reading the measurement surface.

## Pages

| Page | What it covers |
|---|---|
| [Preprocessing and schema](preprocessing.md) | How raw Kaggle and scraped postings become `data/unified.parquet`. |
| [Evidence and sensitivity](evidence.md) | What counts as a robust finding and what remains provisional. |

## The short version

- The preprocessing pipeline creates a single comparable corpus from two Kaggle snapshots and a growing daily scrape.
- `description_core_llm` is the only cleaned text column, and it is coverage-limited.
- `seniority_final` is the primary seniority label, but the YOE proxy must travel with any junior claim.
- `selected_for_llm_frame`, `llm_extraction_coverage`, and `llm_classification_coverage` determine which rows are actually labeled.
- Text claims should be read on the cleaned, labeled subset unless the analysis is explicitly recall-oriented.
