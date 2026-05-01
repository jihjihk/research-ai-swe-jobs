# AI Is Additive to Technology Stacks: AI-Mentioning Postings Require 56% More Technologies

## Headline

AI skill requirements are additive, not substitutive. AI-mentioning postings require **11.4 technologies** on average versus 7.3 for non-AI postings. Overall stack diversity increased from 6.2 to 8.3 mean techs per posting. A new 25-technology AI/ML ecosystem emerged with no 2024 counterpart. Python became the majority language (34.6% to 50.1%).

## Key numbers

| Metric | 2024 | 2026 | Change |
|--------|------|------|--------|
| Mean techs/posting | 6.2 | 8.3 | +2.1 |
| AI-mentioning postings: mean techs | -- | 11.4 | -- |
| Non-AI postings: mean techs | -- | 7.3 | -- |
| Technologies rising (FDR-corrected) | -- | 61 | -- |
| Technologies declining | -- | 12 | -- |
| Technologies stable | -- | 73 | -- |
| Python prevalence | 34.6% | 50.1% | +15.4pp |

## Evidence

### 1. AI is additive, not replacing (T14)

The 146-technology regex dictionary applied to full posting descriptions reveals a clear pattern: postings that mention AI requirements do not substitute away from traditional technologies. Instead, they layer AI technologies on top of existing stacks, resulting in 56% more technology mentions per posting.

This is robust to aggregator exclusion (aggregators slightly inflate non-AI counts) and company capping.

![Technology shift heatmap](../assets/figures/T14/tech_shift_heatmap.png)

### 2. New AI ecosystem emerged from near-zero (T14)

Network analysis of technology co-occurrence patterns reveals a new 25-technology AI/ML community in 2026 with no structural counterpart in 2024:

- **New entrants:** LangChain, RAG, vector databases, agent frameworks, prompt engineering
- **Migrated to AI community:** Python (became the bridge between traditional and AI stacks), TensorFlow, PyTorch

The 2024 technology network had no distinct AI cluster. By 2026, AI/ML formed a coherent community with strong internal co-occurrence and bridges to the traditional stack via Python and cloud infrastructure.

![Co-occurrence network 2026](../assets/figures/T14/cooccurrence_network_2026.png)

### 3. Top risers and decliners (T14)

**Top rising technologies:**

| Technology | 2024 rate | 2026 rate | Change |
|-----------|----------|----------|--------|
| CI/CD | -- | -- | +16.3pp |
| Python | 34.6% | 50.1% | +15.4pp |
| LLM | -- | -- | +12.8pp |
| Agent frameworks | -- | -- | +10.3pp |
| Generative AI | -- | -- | +8.8pp |
| ML | -- | -- | +8.3pp |

**Top declining technologies:**

| Technology | Change |
|-----------|--------|
| Agile | -4.8pp |
| SQL | -3.8pp |
| HTML/CSS | -3.6pp |
| Linux | -2.1pp |
| .NET | -1.8pp |

The decline of Agile methodology language is notable -- an era may be ending as AI-era development practices replace Scrum-oriented workflows.

### 4. Stack diversity DiD (T18)

The cross-occupation DiD for tech stack diversity is +0.78 (SWE vs control), confirming this is a SWE-specific phenomenon. Control occupations did not see comparable technology expansion.

## Sensitivity

- Robust to aggregator exclusion
- Robust to company capping (max 10-20 per company)
- SWE-specific (DiD +0.78 for stack diversity)
- 146-technology dictionary with full-description regex detection
- Within-2024 calibration validates AI technology signals at 5-17x above noise

## Full analysis

- [T14: Technology Ecosystem](../reports/T14.md) -- full technology analysis, co-occurrence networks
- [T18: Cross-Occupation DiD](../reports/T18.md) -- SWE-specificity of stack diversity
- [T12: Text Evolution](../reports/T12.md) -- technology-related text signals
