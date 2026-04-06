# Exploration Task Index

Last updated: 2026-04-05 (Wave 4 complete — all 26 tasks done)

## Seniority recommendation
- **Trend estimation:** Use `seniority_native` (cleanest, 68-83% coverage). Report `seniority_final` as robustness.
- **Cross-sectional:** Use `seniority_final` (best coverage, 81-100%).
- **Coarse:** `seniority_3level` (junior/mid/senior). Associate standalone unusable.
- **Asaniczka:** Exclude from ALL seniority-stratified analyses.
- **Domain stratification:** Stratify by T09 archetype where possible — domain is 10x more important than seniority structurally.

## Column constraints
- **Input file:** `data/unified.parquet`
- **description_core_llm:** Kaggle SWE only (~24K rows). Scraped = 0% (LLM budget forthcoming). Prefer for text analysis; `description_core` is garbage fallback.
- **LLM classification:** ALL NULL. Use rule-based columns.

## Key findings after Gate 3

### What IS strong and SWE-specific (paper core)
1. **AI requirements surge:** 8%→33% (DiD vs control: +24.4pp). Validated as genuine, not aspirational.
2. **Junior share decline:** SWE-specific (DiD = -24.9pp; control junior share *increased*).
3. **Domain recomposition:** ML/AI 4%→27%, Frontend 41%→24%.
4. **AI is additive:** 11.4 vs 7.3 techs per posting. New 25-tech AI community emerged.
5. **YOE slot purification:** 5+ YOE entry: 22.8%→2.4%. Genuine, within-title.
6. **GenAI accelerated 8.3x** between within-2024 and cross-period rates.

### What is NOT SWE-specific or was corrected
7. **Management indicator CORRECTED:** +31pp was measurement error → actual +4-10pp, AND field-wide (DiD ~0).
8. **Soft skills expansion:** SWE grew LESS than control (DiD = -5.1pp).
9. **AI-entry orthogonality:** r=-0.07 (firm), r=-0.04 (metro). Parallel trends, not causal.
10. **57% of aggregate change is compositional** (different companies), not behavioral.
11. **Requirements LAG usage** (~41% vs ~75%), inverting RQ3's hypothesis. Gap narrowing.

### Seniority structure
12. **Associate collapsing toward entry** (relative position 0.30→0.16). 5-level → ~3 tiers.
13. **Senior orchestration shift:** Director management density fell 23%, orchestration surged 46%.
14. **Management expanded at ALL levels** — not migration from senior to entry.

## Task tracking

| Task | Agent | Wave | Status | Key finding | Report |
|------|-------|------|--------|-------------|--------|
| T01 | A | 1 | **done** | 28 cols >50% null; LLM cols null; core viable | [T01.md](T01.md) |
| T02 | A | 1 | **done** | Asaniczka associate NOT junior proxy | [T02.md](T02.md) |
| T03 | B | 1 | **done** | Junior decline 5.9-8.3pp robust across 4 ops | [T03.md](T03.md) |
| T04 | B | 1 | **done** | SWE classification 4-6% FP, <0.5% FN | [T04.md](T04.md) |
| T05 | C | 1 | **done** | Most diffs are artifacts; seniority is exception | [T05.md](T05.md) |
| T06 | C | 1 | **done** | Within-company entry decline -11.8pp | [T06.md](T06.md) |
| T07 | D | 1 | **done** | Well-powered; r>0.97 geographic | [T07.md](T07.md) |
| Prep | Prep | 1.5 | **done** | 52K rows, 146 techs, 384-dim embeddings | — |
| T08 | E | 2 | **done** | YOE slot purification; AI tech 5-17x noise | [T08.md](T08.md) |
| T09 | F | 2 | **done** | Domain > seniority (10x); ML/AI 4%→27% | [T09.md](T09.md) |
| T10 | G | 2 | **done** | AI titles 4x; Staff tripled; titles stable | [T10.md](T10.md) |
| T11 | G | 2 | **done** | Scope inflation (AI +24pp, mgmt +31pp) — **mgmt CORRECTED by T22 to +4-10pp** | [T11.md](T11.md) |
| T12 | H | 2 | **done** | AI dominant text signal; most 2026 terms are boilerplate | [T12.md](T12.md) |
| T13 | H | 2 | **done** | 93.9% of length growth is core content | [T13.md](T13.md) |
| T14 | I | 2 | **done** | 61 rising techs; AI additive; new AI community | [T14.md](T14.md) |
| T15 | I | 2 | **done** | Convergence fails calibration; 7/9 repr-robust | [T15.md](T15.md) |
| T16 | J | 3 | **done** | 57% compositional; AI-entry null (r=-0.07); 4 clusters | [T16.md](T16.md) |
| T17 | J | 3 | **done** | All metros same direction; hub/non-hub null; AI-entry null | [T17.md](T17.md) |
| T18 | K | 3 | **done** | **SWE-SPECIFIC CONFIRMED:** AI DiD +24pp, junior DiD -25pp; mgmt/soft field-wide | [T18.md](T18.md) |
| T19 | K | 3 | **done** | GenAI accelerated 8.3x; within-March stable (CV<10%) | [T19.md](T19.md) |
| T20 | L | 3 | **done** | Boundaries asymmetric: entry/assoc sharp, dir/midsen blurred; associate→entry | [T20.md](T20.md) |
| T21 | L | 3 | **done** | Mgmt migration REJECTED; expanded everywhere; orchestration +46% at director | [T21.md](T21.md) |
| T22 | M | 3 | **done** | Mgmt indicator CORRECTED +4-10pp (not +31pp); AI NOT aspirational (20% vs 30% hedge) | [T22.md](T22.md) |
| T23 | M | 3 | **done** | Requirements LAG usage (~41% vs ~75%); gap narrowing; AI-as-domain may overshoot | [T23.md](T23.md) |
| T24 | N | 4 | **done** | 8 new hypotheses; H1 (domain drives junior decline) most testable; 5 key tensions | [T24.md](T24.md) |
| T25 | N | 4 | **done** | 6 interview artifacts; 3 priority themes; all use corrected findings | [T25.md](T25.md) |
| T26 | N | 4 | **done** | Full synthesis → [SYNTHESIS.md](SYNTHESIS.md) | [T26.md](T26.md) |
