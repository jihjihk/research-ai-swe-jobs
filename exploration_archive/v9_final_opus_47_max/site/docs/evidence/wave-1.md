# Wave 1 — Data Foundation

Data profiling, classification audits, and the seniority-definition panel that underpins every downstream task.

| Task | Owner | Focus |
|---|---|---|
| [T01](tasks/T01.md) | Agent A | Data profile + column coverage — 68,137 SWE LinkedIn rows |
| [T02](tasks/T02.md) | Agent A | Asaniczka associate-as-junior proxy — **not usable** |
| [T03](tasks/T03.md) | Agent B | Seniority label audit — `seniority_final` defensible |
| [T30](tasks/T30.md) | Agent B | **Seniority definition ablation panel — 12/13 direction-consistent** |
| [T04](tasks/T04.md) | Agent B | SWE classification audit — `is_swe` defensible; ML Engineer source-gap |
| [T05](tasks/T05.md) | Agent C | Cross-dataset comparability — `seniority_native` drifts, use YOE |
| [T06](tasks/T06.md) | Agent C | Company concentration + returning-cohort artifact (2,109 firms) |
| [T07](tasks/T07.md) | Agent D | External benchmarks + power analysis — J3/S4 primaries |

## Key Wave 1 findings

- `is_swe` is defensible (zero dual-flag violations, 80.6% rule-LLM agreement).
- `seniority_final` is defensible but LLM abstains on 34 to 53% of SWE rows; director cell accuracy 22 to 27%. Prefer YOE primaries (J3 / S4).
- **Asaniczka has no native entry labels** — use arshkon-only for `seniority_native` sanity checks.
- T30 panel: 7 of 7 junior definitions UP, 5 of 6 senior DOWN; S2 director-only flat.
- Returning-companies cohort (T06) is 2,109 firms, 55% of 2026 postings.
