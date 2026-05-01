# Methodology protocol — SWE labor-market study

**One-page reference. Apply to every claim in `eda/notebooks/findings_consolidated_2026-04-21.ipynb` and downstream paper drafts.**

Date: 2026-04-21. Supersedes ad-hoc per-script choices.

## 1. Text substrate

| Use | Substrate |
|---|---|
| Headline AI-vocab regex matching | **`description_core_llm`** |
| Embedding inputs (BERTopic, sentence-transformers, TF-IDF for clustering) | **`description_core_llm`** with COALESCE fallback to raw `description` |
| Length comparisons (Sv chart) | Both, reported side-by-side |
| Appendix sensitivity disclosure | raw `description`, same regex |

**Reasoning.** `description_core_llm` is the LLM-stripped boilerplate-removed version. Coverage on `unified_core.parquet` is 99.2%. Mean length 2,423 chars vs raw 4,280 — ~42% boilerplate. Boilerplate-driven false positives inflate cross-section AI rates by ~3-5 pp on SWE and 2-12× relative on control. Within-firm and within-pair deltas are substrate-invariant (boilerplate cancels), but level estimates are not.

**Disclosure rule.** Every headline rate cites the core_llm number; the raw equivalent appears in a footnote or appendix table when the two differ by more than 1.5 pp absolute or 25% relative.

## 2. AI-vocab regex

| Use | Pattern |
|---|---|
| Headline matching | **Canonical `AI_VOCAB_PATTERN`** at `eda/scripts/scans.py:50-71` (31 phrases, word-boundary, case-insensitive) |
| Appendix sensitivity disclosure | v9 `ai_strict_v1_rebuilt` (15 phrases, validated 0.96 precision) at `exploration-archive/v9_final_opus_47/artifacts/shared/validated_mgmt_patterns.json` |

**Reasoning.** The canonical pattern includes post-2024 vocabulary (`agentic`, `foundation model`, `mlops`, `ai agent`) that became substantive in 2026 employer language. v9-strict's 0.96 precision is the conservative bound; reporting both shows the regex sensitivity band.

**Do not invent new patterns** for individual analyses. If a finding requires a new pattern, validate precision on a 30-posting hand sample and document in the analysis memo before publishing.

## 3. Sample

| Scope | Choice |
|---|---|
| Primary analysis sample | **`data/unified_core.parquet`** (110k rows, balanced 40/30/30 SWE/adjacent/control, all rows LLM-labeled) |
| Robustness check | Full `data/unified.parquet` (1.45M rows, LinkedIn-only, English, date-flag ok); rates differ <1 pp on every headline tested |
| Period grid | `2024-01`, `2024-04`, `2026-03`, `2026-04` |
| Filter | `description_core_llm IS NOT NULL` for headline scans (99.2% retention) |

## 4. Within-firm panel convention

- **Inclusion:** ≥5 SWE postings in 2024 (kaggle_arshkon ∪ kaggle_asaniczka) AND ≥5 in 2026 (scraped). Returns 292 firms on `unified_core.parquet`.
- **Pair-level test:** same `company_name_canonical × normalized_title` in both periods. Returns smaller panel; report both firm-level and pair-level deltas.

## 5. BERTopic / clustering convention

- Embed `description_core_llm` (COALESCE to `description`) on rows with `llm_extraction_coverage='labeled'` AND length ≥ 200 chars
- **Cap 30 postings per `company_name_canonical × period`** before embedding to prevent prolific employers from dominating cluster centroids
- Sentence-transformers `all-MiniLM-L6-v2`, batches of 256
- BERTopic with UMAP + HDBSCAN as primary; NMF on TF-IDF (k=20) as parallel comparator
- Stability check: 3 seeds, report Adjusted Rand Index between runs

## 6. Self-mention guard for vendor / cluster claims

When a finding cites a vendor token (`openai`, `anthropic`, `claude`, `copilot`, `gemini`, `cursor`, `windsurf`, etc.) or characterizes a cluster by vendor vocabulary, **also report the same number with frontier-AI firms excluded**. The exclusion list:

> OpenAI, Anthropic, Microsoft, Microsoft AI, GitHub, Google, Alphabet, Meta, Facebook, AWS, Amazon, Amazon Web Services, Adobe, NVIDIA, Databricks, xAI, Cohere, Mistral, Hugging Face

**Rationale.** Self-mentions inflate the signal asymmetrically by metro (Bay-headquartered firms over-represent). Tested on the `openai` token: Bay-vs-rest gap collapses from +6.1 pp (raw) to −0.5 pp (under self-mention exclusion). The asymmetric-diffusion finding survives via `agentic` / `ai agent` / `LLM` / `foundation model`, but `openai` is dropped from the headline list.

## 7. Precision-validation discipline

**Any new regex pattern** used for a published claim must be hand-validated on 30 random matches. Patterns scoring below 0.85 precision are rejected. Patterns are pre-committed; do not modify based on results.

This applies retroactively: the v9 management-density finding collapsed under this audit (0.28 precision); the canonical management patterns at `exploration-archive/v9_final_opus_47/artifacts/shared/validated_mgmt_patterns.json` survived.

## 8. Decisions explicitly made; do not relitigate

| Decision | Choice | Source memo |
|---|---|---|
| Drop "Composite C — Who says junior" | Memo only, not published | `composite_C_evaluation.md` |
| Drop `openai` token from Composite A panel 1 | Survives self-mention exclusion only as marginal; story holds via 4 other tokens | `self_mention_audit.md` |
| Drop v9 T34 Applied-AI 15.6× headline | Centroid contamination by 2026-only n-grams | `composite_B_v2.md` + `v9_methodology_audit.md` |
| Lead claim 7 with 2024→2026 delta correlation | More honest than levels; strengthens under core_llm | `claim7_evaluation.md` + `substrate_sensitivity.md` |
| Do not use BLS-derived industry-spread claim as a positive headline | LinkedIn industry composition flat; BLS measures different population | `composite_A_deepdive.md` (DD2 caveat) |

## 9. Appendix tables expected in the paper

For every headline number, the paper appendix should report a 2×2 cell:

|  | Canonical regex | v9 strict |
|---|---|---|
| `description_core_llm` | **Headline** | Sensitivity 1 |
| Raw `description` | Sensitivity 2 | Sensitivity 3 (matches v9 published) |

This is a single 4-cell disclosure per claim, not separate tables.

## Source artifacts

- Substrate audit: `eda/research_memos/substrate_sensitivity.md`
- Regex audit: `eda/research_memos/v9_methodology_audit.md`
- Self-mention audit: `eda/research_memos/self_mention_audit.md`
- Cross-occupation rank audit: `eda/research_memos/claim7_evaluation.md`
- Composite-B BERTopic rerun: `eda/research_memos/composite_B_v2.md`
- Composite-A deep dives: `eda/research_memos/composite_A_deepdive.md`
- Composite-C drop decision: `eda/research_memos/composite_C_evaluation.md`
