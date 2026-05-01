---
piece: 02_copilot_paradox
checker: fact-check
date: 2026-04-21
---

# Fact-check — Piece 02 "The Copilot paradox"

## Filter
`source_platform='linkedin' AND is_swe AND is_english AND date_flag='ok' AND llm_extraction_coverage='labeled'`, `description_core_llm` non-null. Pattern: `(?i)\b(copilot|rag)\b` via DuckDB `regexp_matches`.

## Copilot 2026 scraped
- n=25,547; matches=978; rate = **3.83%**
- Claim: **0.10%**
- **Verdict: DIFFERS MATERIALLY.** Actual rate is ~38× higher than claim. Correct value: **~3.8%** (roughly 1 in 26 postings, not 1 in 1,000).

## RAG 2024 pooled vs 2026 scraped
- 2024 pooled (arshkon + asaniczka): n=22,751; matches=20; rate = **0.088%** (arshkon 0.32%, asaniczka 0.03%)
- 2026 scraped: n=25,547; matches=1,322; rate = **5.17%** (2026-03: 4.69%, 2026-04: 5.59%)
- Ratio: 5.17 / 0.088 = **58.8×**
- Claim: 0.09% → 5.2% (58×)
- **Verdict: MATCHES.** All three numbers land within rounding.

## Caveats
Sampled Copilot and RAG hits are all legitimate (GitHub Copilot, Microsoft/Amazon Copilot, retrieval-augmented generation); `\b`-anchored regex correctly excludes "ragged/fragile" and "copilot" is itself a product name with no common false-positive superstring. The **Copilot headline needs correction**: the lede "0.1% of postings mention Copilot" is wrong by a factor of ~38; the paradox framing should be rebuilt on the actual ~3.8% rate (still far below the 90%-of-Fortune-100 vendor claim, so the paradox survives — just with a less dramatic gap).
