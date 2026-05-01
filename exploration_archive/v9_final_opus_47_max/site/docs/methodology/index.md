# Methodology

This section answers a single question: should you trust the numbers on this site?

A reader landing here cold should start with **[Data sources and samples](data.md)**. That page names the three datasets the study rests on (two public Kaggle releases of LinkedIn postings from 2024, and a fresh 2026 scrape), defines the seven slices of those datasets that individual findings lean on, and sets out what each slice can and cannot support. Most other pages on this site refer back to it.

After that, the rest of the methodology pages can be read in any order:

- **[Preprocessing pipeline](preprocessing.md)**: how the three raw datasets become one analysis-ready table. Eight deterministic cleaning stages, then two stages that send postings to a language model for classification and text cleanup.
- **[LLM prompts](llm-prompts.md)**: the exact text sent to the language model, for anyone who wants to audit what the model was asked to do.
- **[Sensitivity framework](sensitivity.md)**: nine different ways the study stress-tested each headline claim, and the results of those tests. Includes a signal-to-noise calibration built by comparing the two 2024 datasets against each other.
- **[Pattern validation](pattern-validation.md)**: why the study does not trust a regex just because it matches. Every pattern used in a published claim was hand-audited on a 50-posting sample for semantic (not just token-level) precision.
- **[What the paper can claim](claim-scope.md)**: a plain inventory of which claims the evidence supports, which need caveats, and which the data cannot support.
- **[Limitations](limitations.md)**: the known weak spots, grouped by whether they concern the data, the classifiers, the methods, or the external benchmarks.

## Principle

Every claim on this site traces back to a specific analysis and a specific robustness check. Where a claim has a known weakness, the weakness is called out in the same place the claim appears, not hidden at the bottom of a page. The analysis code and underlying tables are reproducible from the repository.
