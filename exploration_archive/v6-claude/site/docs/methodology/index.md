# Methodology

This tab answers the skeptical reader's question: *how was the data built, and can I trust it?*

- **[Data sources](data-sources.md)** — the three posting sources, temporal roles, and known gaps.
- **[Preprocessing pipeline](preprocessing-pipeline.md)** — 10-stage pipeline from raw CSV to analysis parquet, including the two LLM prompts verbatim.
- **[Sensitivity framework](sensitivity-framework.md)** — the four robustness tests every finding must pass.
- **[Limitations and open questions](limitations.md)** — known confounders, coverage gaps, and what the analysis phase needs to address.

## The bottom line

The data pipeline and robustness framework are **not novel in their individual pieces** — DuckDB, pyarrow, BERTopic, Louvain, difference-in-differences, and authorship-style matching are all standard. The contribution is in **combining** them into a framework that a skeptical reviewer can walk through end-to-end on a labor-market corpus where no single source covers the full period.
