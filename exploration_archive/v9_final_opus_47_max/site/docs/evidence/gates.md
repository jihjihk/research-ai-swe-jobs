# Gate memos

Between each wave, an orchestrator-written memo consolidates state, decides pre-commits, and lists hypotheses to induce. Four gates across the exploration.

## Gate timeline

```
   Pre-exploration                Gate 1                  Gate 2               Gate 3
       |                            |                       |                    |
       V                            V                       V                    V
   Wave 0 ----------- Wave 1 ----------- Wave 2 ----- V1 ----------- Wave 3 + 3.5 ----- V2
```

## Gate memos

| Gate | File | Purpose |
|---|---|---|
| 0 | [gate_0_pre_exploration.md](memos/gate_0_pre_exploration.md) | Pre-exploration priors (RQ1 to RQ4), researcher design intent. |
| 1 | [gate_1.md](memos/gate_1.md) | Post-Wave-1 state. Defines T30 J3/S4 primaries + 9 sensitivity pre-commits. |
| 2 | [gate_2.md](memos/gate_2.md) | Post-Wave-2 + V1 state. V1 corrections applied; Wave 3 hypothesis list. |
| 3 | [gate_3.md](memos/gate_3.md) | Post-Wave-3 + Wave-3.5 + V2 state. Unified pre-synthesis memo. **Primary input to SYNTHESIS.md (Wave 4).** |

## Gate 1 pre-commits (eight rules the exploration was held to)

1. J3 (YOE LLM <= 2) + S4 (YOE LLM >= 5) as primary.
2. Pooled-2024 baseline, arshkon-only co-primary for senior claims.
3. Report both magnitudes for senior claims (asaniczka asymmetry).
4. Text-sensitive analyses: labeled-vs-not split on scraped 2026.
5. Entry-specialist exclusion sensitivity for junior claims.
6. Aggregator-exclusion sensitivity for all prevalence claims.
7. Forbid raw-label industry trends (LinkedIn taxonomy drift).
8. Prefer YOE-based proxy over `seniority_native` for temporal claims.

## Gate 3 — the one that drove SYNTHESIS.md

Gate 3 is the single document Agent N (Wave 4 synthesis author) read as primary input. It consolidates:

- Wave 2 discoveries (T08-T15).
- Wave 3 market dynamics (T16-T29).
- Wave 3.5 induced tests (T31-T38).
- V1 + V2 adversarial verifications.

It reaches three decisions:

1. **Paper positioning**: hybrid empirical + methods with A1 + A2 co-lead, supported by A3 and A5.
2. **RQ evolution**: RQ1 reframed (sharpening + universality), RQ3 strengthened (cross-occupation), RQ4 unchanged.
3. **Negative findings foregrounded**: H_B, H_H, H_L, H_N, T29 rejections presented as paper strength.

Read [gate_3.md](memos/gate_3.md) in full for the complete pre-synthesis state.
