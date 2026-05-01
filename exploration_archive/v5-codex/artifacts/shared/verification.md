# Gate 1.5 Shared Artifact Verification

Date: 2026-04-11

## Result

Pass, with one caveat: the shared artifacts are internally consistent, and the embedding index is an exact positional map to the labeled-text subset. Coverage is thin for scraped text, but that is a data limitation already documented in `README.md`, not an artifact failure.

## Checks

- `exploration/artifacts/shared/swe_cleaned_text.parquet`
  - Exists.
  - Row count: `59,972`.
  - `text_source` distribution: `llm = 26,219`, `raw = 33,753`.

- `exploration/artifacts/shared/swe_embeddings.npy`
  - Exists.
  - Shape: `(26,219, 384)`.
  - Dtype: `float32`.

- `exploration/artifacts/shared/swe_embedding_index.parquet`
  - Exists.
  - Row count: `26,219`.
  - `row_index` range: `0..26,218`, with no duplicates.
  - UID set matches the `text_source = 'llm'` rows exactly.
  - Positional comparison against the cleaned-text `llm` rows found `0` mismatches.

- `exploration/artifacts/shared/swe_tech_matrix.parquet`
  - Exists.
  - Row count: `59,972`, matching `swe_cleaned_text.parquet`.

- `exploration/artifacts/shared/company_stoplist.txt`
  - Exists.
  - Nonempty: `60,566` lines.

- `exploration/artifacts/shared/asaniczka_structured_skills.parquet`
  - Exists.
  - Row count: `475,525`.

- `exploration/artifacts/shared/calibration_table.csv`
  - Exists.
  - Row count: `43`.

- `exploration/artifacts/shared/README.md`
  - Exists.
  - Its stated counts and coverage claims match the files checked above.

## Caveats

- `text_source='raw'` still covers most scraped SWE rows, so downstream text-heavy analyses should continue to report labeled counts alongside eligible counts.
- I did not regenerate any artifacts; this is a mechanical consistency check only.
