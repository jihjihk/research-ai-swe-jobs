# The Orthogonality Puzzle: AI Adoption and Junior Decline Are Parallel, Not Causal

## Headline

At the aggregate market level, AI requirements surged and junior hiring declined simultaneously. But within individual companies, these trends are **completely uncorrelated**: firm-level r = -0.07 (p = 0.138), metro-level r = -0.04 (p = 0.850). The paper cannot claim that AI adoption caused junior elimination within firms. Over half (57%) of the aggregate change is compositional -- driven by different companies entering and exiting the market.

## Key numbers

| Level | Correlation | p-value | Interpretation |
|-------|------------|---------|---------------|
| Firm-level (451 companies) | r = -0.07 | 0.138 | Null |
| Metro-level (26 metros) | r = -0.04 | 0.850 | Null |
| Aggregate change: compositional share | 57% | -- | Majority compositional |
| Aggregate change: within-firm share | 43% | -- | Minority behavioral |

## Evidence

### 1. Firm-level orthogonality (T16)

The 451-company overlap panel (companies with 3+ SWE postings in both 2024 and 2026) allows direct measurement of within-firm AI adoption and entry-level changes. Companies that increased their AI mention rates the most did NOT systematically reduce their junior hiring more. The correlation is r = -0.07, statistically indistinguishable from zero.

![AI vs entry share scatter at firm level](../assets/figures/T16/ai_vs_entry_scatter.png)

### 2. Metro-level orthogonality (T17)

Aggregating to the metro level (26 metros) produces the same null result: r = -0.04 (p = 0.850). All 26 metros show positive changes on all five metrics (entry share decline, AI prevalence increase, org scope increase, description length increase, tech diversity increase), but the magnitudes are uncorrelated. The transformation is nationally uniform but the AI and entry-level components vary independently.

![AI vs entry share at metro level](../assets/figures/T17/ai_vs_entry_metro.png)

### 3. Shift-share decomposition: 57% compositional (T16)

The shift-share decomposition reveals that more than half of the aggregate change reflects different companies posting, not the same companies changing behavior:

- **New market entrants** (2026 only): 24.3% AI rate -- arrived already AI-forward
- **Market exits** (2024 only): 2.5% AI rate -- departed with low AI adoption
- **Overlap companies:** showed real within-firm increases, but these account for only 43% of the total

![Decomposition of aggregate changes](../assets/figures/T16/decomposition_bars.png)

### 4. Four company clusters (T16)

The overlap panel reveals four distinct company strategies:

| Cluster | Share | Behavior |
|---------|-------|----------|
| Stable Traditional | 55% | Low AI adoption, modest changes |
| AI Transformers | 11% | Large AI adoption increase, existing companies pivoting |
| New AI-Native | 22% | Arrived with high AI rates (2026 entrants) |
| Declining Legacy | 12% | Reduced posting volume, low AI |

The compositional dominance comes from the New AI-Native cluster: these companies did not exist in the 2024 sample and arrived with fundamentally different posting profiles.

![Company cluster heatmap](../assets/figures/T16/cluster_heatmap.png)

## What this means for the paper

The orthogonality finding is the most important causal caveat in the study. It constrains what the paper can and cannot claim:

**Can claim:**
- AI requirements surged in SWE postings (SWE-specific, genuine)
- Entry-level SWE share declined (SWE-specific, robust)
- These are parallel market-level trends

**Cannot claim:**
- AI adoption caused junior elimination within firms
- Companies that adopted AI reduced junior hiring
- The aggregate pattern reflects within-firm restructuring (57% is compositional)

## Possible explanations (for interviews)

The orthogonality puzzle is the top priority for practitioner interviews (RQ4). Candidate explanations:

1. **Separate mechanisms:** AI adoption and junior hiring decisions are made by different organizational units (engineering vs HR/headcount planning) responding to different pressures
2. **Market equilibrium:** AI-driven productivity gains reduce aggregate junior demand at the market level, but individual firms don't internalize this -- the effect operates through reduced hiring budgets set by macroeconomic conditions
3. **Time-lag causation:** Our two snapshots cannot capture a lagged effect. AI adoption in 2024-2025 may affect junior hiring in 2027+
4. **Genuine independence:** The junior decline may be driven by factors unrelated to AI (e.g., domain recomposition, economic conditions, post-pandemic hiring normalization)

## Geographic uniformity (T17)

All 26 metros show the same directional changes. Tech hubs vs non-hubs show no significant differences (all p > 0.08). The transformation is nationally uniform -- this is not a Bay Area phenomenon spreading to other metros.

![Metro-level heatmap](../assets/figures/T17/metro_heatmap.png)

## Sensitivity

The orthogonality finding is a **negative result** -- it must be shown to not appear under any specification:

- Holds across panel size thresholds (min 3, 5, 10 postings per company)
- Holds across seniority operationalizations
- Holds with and without aggregator postings
- Holds at both firm and metro aggregation levels
- r is consistently near zero (range: -0.04 to -0.07) across all specifications

## Full analysis

- [T16: Company Strategies](../reports/T16.md) -- firm-level orthogonality and shift-share decomposition
- [T17: Geographic Structure](../reports/T17.md) -- metro-level orthogonality and geographic uniformity
- [T24: New Hypotheses](../reports/T24.md) -- candidate explanations and testable hypotheses
- [T25: Interview Artifacts](../reports/T25.md) -- interview probes for the orthogonality puzzle
