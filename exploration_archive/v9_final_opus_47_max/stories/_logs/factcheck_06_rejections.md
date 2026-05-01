# Fact-check: Piece 06 "The rejection of the easy explanation"

**Date:** 2026-04-21 · **Subagent:** Opus 4.7 max-reasoning

## Claim verification table

| # | Claim | Source | Verified | Flags |
|---|---|---|---|---|
| 1 | LLM-authorship REJECTED (T29): 80-130% preserved on low-LLM Q1; length 52% | `exploration/reports/T29.md` §7 | YES | T29 §7 table confirms: ai_binary 0.88×, scope 1.28×, tech_count 0.97×, credential_stack J3 0.97×, breadth_resid 0.80×; length ratio 0.52 exact. Preservation band 80-128% matches "80-130%". |
| 2 | Hiring-bar REJECTED (T33): \|ρ\|≤0.11; 0/50 loosening | `exploration/reports/T33.md` §3, §6 | PARTIAL | T33 headline says \|ρ\|≤0.28 (not 0.11). Under T13 classifier, all 4 proxy correlations are \|ρ\|≤0.088 (matches "≤0.11"). Simple-regex hits +0.19-0.28. 0/50 loosening language exact match. Draft should say "≤0.11 under T13" or relax bound to "≤0.28". |
| 3 | Hiring-selectivity REJECTED (T38): \|r\|<0.11; desc_len r=+0.20 | `exploration/reports/T38.md` §2 | YES | T38 §2 primary matrix (n=243): breadth r=−0.032, ai_strict r=−0.089, scope r=−0.033, mentor r=−0.072, YOE r=−0.008 — all \|r\|<0.11. desc_len r=+0.203 p=0.0015 exact. |
| 4 | Sampling-frame REJECTED (T37): 13-14/15 robust | `exploration/reports/T37.md` §2 | YES | T37 headline table: 14 of 15 ratio ≥0.80 (only H_d-J3 at 0.70); T37 text says "13 of 15" in §TL;DR but saved table shows 14/15. V2 §1 flag #3 confirms this self-audit discrepancy. Draft's "13-14" framing correctly captures both numbers. |
| 5 | Legacy→AI REJECTED (T36): 2026 neighbors 3.6% vs market 14.4% | `exploration/reports/T36.md` §3 | YES | T36 ai_vocab_comparison: 6 neighbors 1.0-7.1%, mean 3.6%; market rate 14.4% (14.36% per V2 §1). Exact match. |

## Overall verdict: 5/5 VERIFIED (one partial on exact bound)

## Revision needed

Claim 2's "\|ρ\|≤0.11" is narrower than T33's headline "\|ρ\|≤0.28" (which covers simple-regex classifier). Under the primary T13 classifier alone the bound ≤0.11 holds (max 0.088). Either narrow the draft to "under T13 classifier, all \|ρ\|≤0.09" or loosen to T33's native "\|ρ\|≤0.28 mixed-sign across 2 classifiers".

## V2 flags

Only T37 discrepancy (13 vs 14 of 15) is V2-flagged; piece 06 already hedges "13-14". No V2 flag on T29, T33, T36, T38 rejection claims.

## One-sentence summary

All five rejection claims survive fact-check; only the T33 correlation bound needs a classifier qualifier to stay rigorous.
