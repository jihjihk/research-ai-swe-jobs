# Shared Analytical Artifacts -- Wave 1.5

Built: 2026-04-05 20:27:32
Total build time: 1217.4s (20.3m)

## Contents

### swe_cleaned_text.parquet
- **Rows:** 52327
- **Columns:** uid, description_cleaned, text_source, source, period, seniority_final, seniority_3level, is_aggregator, company_name_canonical, metro_area, yoe_extracted, swe_classification_tier, seniority_final_source
- **Text source distribution:**
  - llm: 23815 rows (45.5%)
  - rule: 28512 rows (54.5%)
  - raw: 0 rows (0.0%)
- **Build time:** 23.1s
- **Description:** Best-available description text for each SWE LinkedIn row. Priority: description_core_llm (where llm_extraction_coverage='labeled') > description_core > description. Each row's own company_name_canonical is stripped as a complete phrase (not individual tokens), and English stopwords are removed. This approach preserves technology terms like "machine learning" that happen to appear in some company names.

### company_stoplist.txt
- **Tokens:** 9422
- **Description:** Unique tokens (3+ chars) from company_name_canonical values across SWE rows. Excludes generic English words and numeric-only tokens. Used during text cleaning.

### swe_embeddings.npy
- **Shape:** 52327 x 384 (float32)
- **Model:** all-MiniLM-L6-v2 (sentence-transformers)
- **Build time:** 958.6s
- **Description:** Sentence-transformer embeddings of description_cleaned (first ~3000 chars, ~512 tokens). Row order matches swe_embedding_index.parquet.

### swe_embedding_index.parquet
- **Rows:** 52327
- **Columns:** row_index (int32), uid (string)
- **Description:** Maps embedding matrix row index to uid. Join with other artifacts on uid.

### swe_tech_matrix.parquet
- **Rows:** 52327
- **Columns:** uid + 146 boolean technology columns
- **Build time:** 235.1s
- **Description:** Binary matrix of technology mentions detected via regex in description_cleaned. Covers languages, frameworks, cloud/devops, data, AI/ML, AI tools, testing, practices, mobile, and security.

### asaniczka_structured_skills.parquet
- **Source rows:** 23192
- **Total skill mentions:** 609894
- **Unique skills:** 122533
- **Build time:** 0.4s
- **Columns:** uid (string), skill (string)
- **Description:** Parsed comma-separated skills from asaniczka SWE rows. Long format (one row per uid-skill pair).

## Filters Applied

All artifacts use the default SQL filters:
```sql
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
```

## Known Issues / Partial Coverage

- **Text quality varies by source:** LLM-cleaned text covers ~45% of rows (Kaggle with llm_extraction_coverage='labeled'). Remaining rows use rule-based boilerplate removal (~44% accuracy) or raw text. Filter on text_source='llm' for highest quality text subset.
- **Scraped data has NO LLM-cleaned text** -- all ~24K scraped rows use rule-based or raw fallback.
- **Company name removal** strips each row's own company_name_canonical as a complete phrase, not individual tokens. The company_stoplist.txt is retained as a reference artifact but is NOT used for text cleaning (earlier approach of token-level stripping destroyed tech terms like "machine learning").
- **Technology regex patterns:** 'go' is restricted to clear programming contexts (golang, "experience with go", etc.) to avoid false positives. Some patterns like 'rust', 'swift', 'flask' may have minor false-positive rates in non-tech contexts.
- **ai_rag** pattern may match the standalone word "rag" in non-AI contexts; rates should be low in SWE job postings.

## Quick Reference: Top Tech by Period

| Technology | 2024-01 | 2024-04 | 2026-03 |
|---|---|---|---|
| lang_python | 34.5% | 35.2% | 50.1% |
| cloud_aws | 29.2% | 28.0% | 36.1% |
| practice_agile | 32.8% | 29.5% | 27.4% |
| ai_llm | 1.6% | 1.6% | 14.4% |
| ai_generative_ai | 0.9% | 1.6% | 9.8% |
| ai_agent_frameworks | 0.0% | 0.1% | 10.3% |
| ai_rag | 0.0% | 0.4% | 5.6% |
| ai_prompt_eng | 0.2% | 0.3% | 3.8% |
