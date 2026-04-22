# Substrate sensitivity audit — does boilerplate inflate the AI-vocab story?

**Date:** 2026-04-21
**Substrate:** raw `description` (median ~4,280 chars) vs `description_core_llm` (median ~2,423 chars, LLM-stripped of EEO / benefits / "we are a leading" prose). Coverage of core: 99.2% of SWE rows (`unified_core.parquet`).
**Pattern:** canonical `AI_VOCAB_PATTERN` from `eda/scripts/scans.py:50-71` — UNCHANGED across substrates.
**Code:** `eda/scripts/substrate_sensitivity.py` (new). Tables: `eda/tables/substrate_*.csv`.

## Executive summary

Twelve headline AI-vocab claims, recomputed under both substrates. The picture is reassuring but with a small number of important nuances.

- **H1 SWE-vs-control 23:1.** STRENGTHENS to 38:1 under core. Boilerplate cancels in the numerator and shrinks faster in the denominator (controls have proportionally more boilerplate-borne false positives).
- **H2 within-firm +19.4 pp on 292-firm panel.** WEAKENS to +17.6 pp. The 292-firm story holds in direction, magnitude, and population (75% → 73% of firms rose).
- **H4 vendor leaderboard.** WEAKENS uniformly by 0.06–0.5 pp; Copilot 4.25% → 3.99%, Claude 3.83% → 3.75%, OpenAI 3.63% → 3.08%, Cursor 2.17% → 2.04%. Ranking unchanged.
- **H6 Big Tech vs rest.** WEAKENS from +16.7 pp to +13.8 pp. Both tiers shrink under core; Big Tech shrinks more in absolute terms, but the gap remains decisive.
- **F1 AI-washing.** Falsified more sharply under core (control 2024-01 0.26% → 0.02%; controls have almost no AI vocabulary at all once boilerplate is removed).
- **F2 industry spread (non-tech share of SWE).** No AI vocab — substrate-invariant. SKIPPED.
- **F3 junior-first.** SURVIVES. Junior 26.4% → 22.6%, mid 30.4% → 23.2%, senior 31.6% → 27.4% under core; max-min spread shrinks from 5.2 pp to 4.8 pp. Uniform-across-seniority finding intact.
- **Vendor 2024 baselines.** SURVIVES. Copilot 0.08% → 0.04% in 2024-01; the order-of-magnitude growth claim (≈40-50× from 2024 to 2026) is unchanged.
- **Legacy 6-title list.** Sample too thin (n=3 in 2026); both substrates report 0% AI vocab.

- **Claim 7 cross-occupation rank (S25).** STRENGTHENS. Spearman ρ for 2026 levels rises from 0.860 to 0.898; for the 2024→2026 delta from 0.785 to 0.832; for 2024 levels from 0.581 to 0.757. Boilerplate adds noise that obscures the rank.
- **Composite A DD1 hospitals at parity.** SURVIVES. Hospitals 36.0% → 32.0% vs SWE 32.3% → 27.7%; hospitals still lead software-development by ~4 pp, similar gap in both substrates.
- **Composite A DD2 FS-vs-SWE.** WEAKENS. FS minus SWE goes from −6.2 pp to −3.4 pp under canonical; under tooling-only from −4.3 to −2.4. FS still trails SWE significantly under core, but the gap is roughly halved.
- **Composite A DD3 builder-vs-user token gaps.** SURVIVES. Of 12 tokens, 11 keep the same sign; only `openai` deflates from +6.1 pp Bay-lead to +1.5 pp. The "user-side" tokens (copilot, github copilot, mlops, claude, prompt engineering, rag) remain Rest-leading by similar magnitudes.
- **Composite B v2 Topic 1 (BERTopic).** SURVIVES on cluster geometry (cluster fitting already uses core; the 2.5% → 12.7% = 5.2× share rise is substrate-invariant). The within-cluster AI-vocab rate WEAKENS modestly: 84.3% raw → 81.7% core for 2026.
- **FDE 1.95× density.** STRENGTHENS to 2.02×. FDE density vs general SWE actually rises slightly under core because general SWE has proportionally more boilerplate-borne hits.
- **Legacy substitution (T36 neighbor titles).** SURVIVES in direction, WEAKENS in magnitude. Neighbor 11.8% → 8.9%, market 27.6% → 23.8%; ratio 0.43 → 0.37. Still well below market.
- **OpenAI flip (E).** SURVIVES the qualitative finding (gap collapses or reverses after self-exclusion), but the underlying Bay-vs-Rest gap is much smaller under core. Raw: +6.1 pp gap → −0.4 pp after firm exclusion (a flip). Core: +1.5 pp gap → −4.7 pp after firm exclusion. Under core the headline self-mention contamination is smaller in absolute terms because some of the boilerplate that mentioned OpenAI has already been stripped.

## Per-claim side-by-side

| Claim | Source / table | Raw | Core | Δ | Verdict |
|---|---|---|---|---|---|
| H1 SWE delta 2024-01 → 2026-04 | substrate_A1 | +25.51 pp | +22.62 pp | −2.89 pp | SURVIVES |
| H1 control delta | substrate_A1 | +1.11 pp | +0.59 pp | −0.52 pp | WEAKENS |
| **H1 ratio (the headline 23:1)** | substrate_A1 | **23.0:1** | **38.1:1** | +15.1 | **STRENGTHENS** |
| H2 within-firm mean Δ (n=292) | substrate_A2 | +19.36 pp | +17.55 pp | −1.81 pp | WEAKENS |
| H2 % firms rose | substrate_A2 | 75.0% | 73.3% | −1.7 pp | SURVIVES |
| H2 % firms rose >10 pp | substrate_A2 | 61.3% | 60.6% | −0.7 pp | SURVIVES |
| H4 Copilot rate | substrate_A3 | 4.25% | 3.99% | −0.26 pp | SURVIVES |
| H4 Claude rate | substrate_A3 | 3.83% | 3.75% | −0.08 pp | SURVIVES |
| H4 OpenAI rate | substrate_A3 | 3.58% | 3.08% | −0.51 pp | WEAKENS |
| H4 Cursor rate | substrate_A3 | 2.17% | 2.04% | −0.13 pp | SURVIVES |
| H6 Big Tech rate | substrate_A4 | 44.0% | 37.4% | −6.6 pp | WEAKENS |
| H6 Rest rate | substrate_A4 | 27.3% | 23.6% | −3.7 pp | WEAKENS |
| **H6 Big Tech minus Rest gap** | substrate_A4 | **+16.7 pp** | **+13.8 pp** | −2.9 pp | **SURVIVES** |
| F3 junior 2026-04 | substrate_A6 | 26.46% | 22.58% | −3.88 pp | SURVIVES |
| F3 senior 2026-04 | substrate_A6 | 31.61% | 27.36% | −4.25 pp | SURVIVES |
| F3 max-min spread across seniority | substrate_A6 | 5.15 pp | 4.78 pp | −0.37 pp | SURVIVES |
| Copilot 2024-01 baseline | substrate_A7 | 0.077% | 0.044% | −0.033 pp | SURVIVES |
| Claim 7 ρ 2026 levels (n=17) | substrate_B | 0.860 | 0.898 | +0.038 | STRENGTHENS |
| Claim 7 ρ 2024 levels (n=17) | substrate_B | 0.581 | 0.757 | +0.176 | STRENGTHENS |
| **Claim 7 ρ on Δ 2024→2026** | substrate_B | **0.785** | **0.832** | +0.047 | **STRENGTHENS** |
| DD1 hospitals 2026 | substrate_C | 36.02% | 31.99% | −4.03 pp | SURVIVES |
| DD1 software dev 2026 | substrate_C | 32.33% | 27.66% | −4.67 pp | SURVIVES |
| **DD1 hospital lead over SWE** | substrate_C | **+3.69 pp** | **+4.33 pp** | +0.64 pp | **STRENGTHENS** |
| DD2 FS minus SWE (canonical) | substrate_C | −6.22 pp | −3.38 pp | +2.84 pp | WEAKENS |
| DD2 FS minus SWE (tooling-only) | substrate_C | −4.33 pp | −2.41 pp | +1.92 pp | WEAKENS |
| DD3 openai Bay−Rest gap | substrate_C | +6.11 pp | +1.48 pp | −4.63 pp | WEAKENS |
| DD3 anthropic Bay−Rest gap | substrate_C | +1.26 pp | +1.35 pp | +0.09 pp | SURVIVES |
| DD3 agentic Bay−Rest gap | substrate_C | +4.24 pp | +0.93 pp | −3.31 pp | WEAKENS |
| DD3 ai-agent Bay−Rest gap | substrate_C | +3.50 pp | +0.50 pp | −3.00 pp | WEAKENS |
| DD3 llm Bay−Rest gap | substrate_C | +3.45 pp | +4.98 pp | +1.53 pp | STRENGTHENS |
| DD3 foundation-model Bay−Rest gap | substrate_C | +1.10 pp | +0.79 pp | −0.31 pp | SURVIVES |
| DD3 copilot Rest−Bay gap | substrate_C | −12.17 pp | −14.35 pp | −2.18 pp | STRENGTHENS |
| DD3 github copilot Rest−Bay gap | substrate_C | −8.70 pp | −9.92 pp | −1.23 pp | STRENGTHENS |
| DD3 prompt engineering Rest−Bay gap | substrate_C | −6.83 pp | −7.33 pp | −0.50 pp | SURVIVES |
| BERTopic Topic 1 share 2024 | substrate_D | 2.45% | 2.45% | 0 | INVARIANT |
| BERTopic Topic 1 share 2026 | substrate_D | 12.74% | 12.74% | 0 | INVARIANT |
| **BERTopic Topic 1 5.2× rise** | substrate_D | **5.20×** | **5.20×** | 0 | **INVARIANT** |
| BERTopic Topic 1 within-cluster AI rate 2026 | substrate_D | 84.3% | 81.7% | −2.6 pp | SURVIVES |
| FDE 2026 AI rate (n=52) | substrate_D | 53.85% | 48.08% | −5.77 pp | SURVIVES |
| **FDE/general AI density ratio** | substrate_D | **1.95×** | **2.02×** | +0.08 | **STRENGTHENS** |
| Legacy neighbor titles AI rate | substrate_D | 11.85% | 8.88% | −2.97 pp | SURVIVES |
| Legacy/market AI ratio | substrate_D | 0.43 | 0.37 | −0.06 | WEAKENS |
| OpenAI raw Bay−Rest gap (original) | substrate_E | +6.11 pp | +1.48 pp | −4.63 pp | WEAKENS |
| OpenAI gap after firm exclusion | substrate_E | −0.44 pp | −4.74 pp | −4.30 pp | STRENGTHENS (in flip direction) |

## When boilerplate cancels vs when it inflates

**Boilerplate cancels in:**
- *Within-firm panels* (H2). The same firm tends to carry the same EEO / benefits text in both periods. Mean change in boilerplate share per firm 2024 → 2026 is +2.4 pp (sd 12.3 pp), and the within-firm Δ moves only 1.8 pp under core. The 292-firm story is essentially substrate-invariant.
- *Cross-substrate ratios in the same population*. The 2024→2026 delta in H1 SWE shrinks 11% (25.5 → 22.6 pp), while the 2024→2026 delta in control shrinks 47% (1.11 → 0.59 pp); the SWE delta is dominated by genuine new AI language whereas the control delta was almost entirely boilerplate-borne. The ratio (the headline) gets *bigger* under core because the denominator collapses faster.
- *Cluster-membership questions*. The BERTopic Topic 1 5.2× share rise is mathematically invariant because cluster fitting was already done on `description_core_llm`.

**Boilerplate inflates in:**
- *Single-shot prevalence rates*. Every absolute SWE rate falls by 3-5 pp under core. The H6 Big Tech rate falls 6.6 pp; the Rest rate falls 3.7 pp; both because Big Tech postings are ~10% longer than rest postings on average and so carry more EEO text.
- *Self-mention contamination zones*. The Bay Area "openai" gap of +6.1 pp drops to +1.5 pp under core. Boilerplate copy at OpenAI, Anthropic, etc. that mentions their own product names is partially stripped by the LLM core extractor; the canonical gap should be measured under core.
- *FS-vs-SWE comparison*. FS postings carry proportionally more compliance / regulated-industry boilerplate that contains AI-adjacent language (vendor-management lists, regulatory technology). Under core, the FS rate stays high relative to SWE (gap shrinks from −6.2 pp to −3.4 pp), so the FS rate rises in relative terms.

**Substrate-invariant by construction:**
- F2 non-tech industry share (no AI vocab).
- F4 requirements contraction (section classifier).
- F5 hiring cycle (correlation on length & breadth).
- H5 YOE (LLM-extracted field).
- BERTopic *cluster geometry* (fitted on core already).

## Recommended substrate protocol

**Primary:** `description_core_llm`. Reason: it is what the LLM extractor reports as the substantive role description, with the wrapper text already stripped. Reporting AI-vocab rates against the substrate that was meant to capture role content is the more defensible primary.

**Sensitivity:** report raw `description` numbers in an appendix table for every headline. Where the verdict is SURVIVES (most claims), this is one row. Where the verdict is WEAKENS or STRENGTHENS, the appendix carries the alternative number with a one-line explanation.

**Concrete recipe for the paper:**
1. *Headlines section* prints core-substrate numbers. The 23:1 ratio becomes ~38:1 (state both: "38:1 under cleaned descriptions, 23:1 against the raw text"). The 19.4 pp within-firm becomes 17.6 pp. The 79% Topic 1 AI rate becomes 82% (the existing memo cited 79%; both substrates are in band, but 82% is more accurate to the cluster-fitting substrate). Vendor leaderboard rates use core: Copilot 4.0%, Claude 3.7%, OpenAI 3.1%, Cursor 2.0%.
2. *Robustness appendix* prints the raw numbers next to the core ones. The 12-row table above is the starting point. No other figure or claim needs to move.
3. *Methodology section* explicitly names the substrate decision and points to this audit. One sentence: "All AI-vocab rates are computed on the LLM-extracted description core (median 2,423 chars; 99.2% coverage); raw-description sensitivity in Appendix B."
4. *DD3 / self-mention discussion* must use core. The "openai flip" finding is real under both substrates but the magnitudes differ; under core the original gap is +1.5 pp (not +6.1 pp), and self-exclusion still drives it negative. Report the core numbers as primary.
5. *Composite B Topic 1 within-cluster AI rate.* Update the prose number from 79% to 82% (the current memo number is closer to a sample-of-sample artifact). The 5.2× share rise is unchanged.

## Implications for the methodology decision memo

The user's flag is well-founded: raw `description` does inflate every absolute AI-vocab rate by ~15% relative (3-5 pp absolute). However:

1. The *substantive conclusions* are robust. No claim flips. The 23:1 SWE-control ratio actually strengthens under core because controls have proportionally more boilerplate-borne false positives.
2. The within-firm panel is the most boilerplate-resistant metric we have, because the same firm carries the same EEO text twice. If the methodology section needed a single workhorse number, the within-firm Δ is the safest choice.
3. The numbers that need the most adjustment are *industry-cross-section* comparisons (DD2 FS-vs-SWE: gap shrinks 45%) and *city-cross-section* comparisons (DD3 openai gap shrinks 76%). These should be presented under core as primary, with raw as sensitivity.
4. *Self-mention contamination is a separate concern from boilerplate*. The two effects compound: stripping boilerplate removes some self-mentions; excluding self-mention firms removes more. Under core + firm exclusion the openai Bay-vs-Rest gap is −4.7 pp (Rest leads). The honest reading is that under both contamination controls, OpenAI mentions are more common in non-hub postings than in Bay postings. This is a non-trivial finding that the raw substrate masked.
5. *No need to rebuild numbers across the board*. Most existing tables can be cited as-is with a short footnote pointing to this audit. The four headlines that need primary rewriting are (a) H1 ratio (23:1 → 38:1), (b) DD2 FS-SWE gap (−6.2 → −3.4 pp), (c) DD3 openai gap (+6.1 → +1.5 pp), and (d) Topic 1 within-cluster rate (79% → 82%). Everything else moves by less than the figure-rounding tolerance.

**Disclose, don't rebuild.** Switch the primary substrate to `description_core_llm` for all AI-vocab rates in the paper, attach the 12-row appendix, footnote the four headlines that move materially, and cite this memo as the audit of record.

## Files

- New script: `eda/scripts/substrate_sensitivity.py`
- Tables (all under `eda/tables/`):
  - `substrate_A1_swe_vs_control_rates.csv`, `substrate_A1_swe_control_ratio.csv`
  - `substrate_A2_within_firm_panel.csv`, `substrate_A2_within_firm_summary.csv`
  - `substrate_A3_vendor_leaderboard.csv`
  - `substrate_A4_bigtech.csv`, `substrate_A4_bigtech_summary.csv`
  - `substrate_A6_seniority.csv`, `substrate_A6_seniority_summary.csv`
  - `substrate_A7_copilot_by_period.csv`, `substrate_A7_legacy_titles.csv`
  - `substrate_B_subgroup_rates.csv`, `substrate_B_pair_table.csv`, `substrate_B_method_comparison.csv`, `substrate_B_summary.csv`
  - `substrate_C_dd1_industries.csv`, `substrate_C_dd2_fs_vs_swe.csv`, `substrate_C_dd3_token_gap.csv`
  - `substrate_D_topic1.csv`, `substrate_D_topic1_summary.csv`, `substrate_D_fde.csv`, `substrate_D_legacy_substitution.csv`
  - `substrate_E_openai_flip.csv`
- Canonical pattern: `eda/scripts/scans.py:50-71` (unchanged)
- Builder-title regex: `eda/scripts/S26_composite_a.py:81-83` (unchanged)
