# AI Requirements Surged +24pp (DiD), Validated as Genuine Hiring Requirements

## Headline

AI competency requirements in SWE job postings surged from 8% to 33% between 2024 and 2026. This is massively SWE-specific: control occupations moved from 1% to 3%. The difference-in-differences estimate is **+24.4 percentage points**. The requirements are genuine hiring bars, not aspirational language.

## Key numbers

| Metric | 2024 | 2026 | Change |
|--------|------|------|--------|
| SWE AI mention rate (broad) | 7.6% | 33.2% | +25.6pp |
| Entry-level AI mention rate | 3.9% | 27.5% | +23.6pp |
| Control occupation AI rate | 1.2% | 3.6% | +2.4pp |
| **DiD (SWE vs control)** | -- | -- | **+24.4pp** |
| GenAI-specific DiD | -- | -- | +19.4pp |

## Evidence

### 1. SWE-specific, not a field-wide trend (T18)

The cross-occupation difference-in-differences analysis (T18) compared SWE postings against ~142K control-occupation postings. AI/ML prevalence increased by +26.7pp in SWE versus only +2.4pp in control occupations, yielding a DiD of +24.4pp. SWE-adjacent occupations (data analysts, IT managers) tracked SWE closely at ~39% AI rate, confirming the meaningful divide is technical vs non-technical roles.

![AI adoption gradient across occupation types](../assets/figures/T18/ai_adoption_gradient.png)

### 2. Genuine, not aspirational (T22)

The aspiration-ratio analysis (T22) tested whether AI requirements are "ghost" language -- aspirational terms that don't function as real hiring screens. The hedge fraction (proportion of requirements softened with "preferred," "nice to have," etc.) tells the story:

| Period | AI hedge fraction | Non-AI hedge fraction |
|--------|------------------|-----------------------|
| 2024-01 | 40.2% | 34.2% |
| 2024-04 | 30.9% | 21.9% |
| 2026-03 | **20.0%** | **30.0%** |

AI requirements reversed from more-aspirational-than-average (2024) to less-aspirational-than-average (2026). Employers who list AI requirements mean it.

![AI aspiration comparison](../assets/figures/T22/ai_aspiration_comparison.png)

### 3. Within-2024 calibration validates the signal (T08)

The within-2024 calibration (comparing arshkon April 2024 vs asaniczka January 2024) found AI technology signals at 5-17x above the noise floor. The AI surge is not an instrument artifact.

## Specific tool adoption

While broad AI mention rates are high, specific tool mentions are more modest:

- LLM (general): 10%
- GitHub Copilot: 4%
- Claude: 4%
- LangChain: emerging from near-zero
- RAG / vector DBs / agent frameworks: new 25-technology AI ecosystem with no 2024 counterpart (T14)

This suggests employers are requiring general AI literacy rather than specific tool expertise.

## GenAI acceleration

GenAI adoption accelerated 8.3x between the within-2024 rate (+1.2pp/year) and the cross-period rate (+10.2pp/year), consistent with the wave of model releases (GPT-4o, Claude 3.5, o1, DeepSeek V3) that occurred between our snapshots (T19).

![GenAI temporal growth](../assets/figures/T19/rate_of_change.png)

## Sensitivity

- Robust to aggregator exclusion (aggregators are 12-27% of sample)
- Robust to company capping (max 10-20 per company)
- Robust to SWE classification tier (regex-only vs full sample)
- Within-2024 calibration ratio: 5-17x above noise

## Full analysis

- [T18: Cross-Occupation DiD](../reports/T18.md) -- SWE-specificity confirmation
- [T22: Ghost Forensics](../reports/T22.md) -- aspiration-ratio validation
- [T08: Distribution Profiling](../reports/T08.md) -- within-2024 calibration
- [T14: Technology Ecosystem](../reports/T14.md) -- specific tool adoption rates
- [T19: Temporal Patterns](../reports/T19.md) -- GenAI acceleration
