# Entry-Level SWE Share Declined While Control Occupations Hired More Juniors (DiD = -25pp)

## Headline

Entry-level SWE posting share declined from 22.3% to 14.0% between 2024 and 2026, while control occupations' junior share **increased**. The difference-in-differences estimate is **-24.9 percentage points**. The decline is accompanied by "slot purification": the surviving entry-level postings are more genuinely junior (lower YOE, more readable, narrower experience bands).

## Key numbers

| Metric | 2024 | 2026 | Change |
|--------|------|------|--------|
| SWE entry share (seniority_native) | 22.3% | 14.0% | -8.3pp |
| Control junior share | -- | -- | +21.9pp (increased) |
| **DiD (SWE vs control)** | -- | -- | **-24.9pp** |
| Within-company entry decline | 25.0% | 13.2% | -11.8pp |
| 5+ YOE entry-level postings | 22.8% | 2.4% | -20.4pp |
| Median entry YOE | 3.0 | 2.0 | -1.0 |
| Entry-level FK readability | 16.5 | 15.3 | -1.2 (easier) |

## Evidence

### 1. SWE-specific, opposite to control occupations (T18)

The cross-occupation DiD (T18) is the strongest evidence. SWE entry share declined while control occupations' junior share moved in the opposite direction. This divergence (-24.9pp DiD) cannot be explained by macroeconomic factors or general labor market trends -- something SWE-specific is happening.

![Parallel trends across occupations](../assets/figures/T18/parallel_trends.png)

### 2. Within-company decline exceeds aggregate (T06)

Among 110 companies appearing in both periods (T06), entry share dropped from 25.0% to 13.2% (-11.8pp). This is stronger than the aggregate decline, meaning compositional effects actually dampen the signal. New 2026 market entrants have slightly higher junior share (15.9%) than returning companies (13.2%). The decline is not a composition artifact for this measure -- it is real restructuring within established companies.

![Company-level junior share trends](../assets/figures/T25/junior_share_trend.png)

### 3. YOE slot purification (T08)

Five independent decompositions (T08) confirm that high-experience "entry-level" postings nearly vanished:

- **By YOE band:** 5+ YOE entry dropped from 22.8% to 2.4%
- **By title:** Consistent within individual titles (not driven by title mix)
- **By company type:** Holds for both overlap-panel and new-entrant companies
- **By aggregator status:** Holds with and without aggregator postings
- **By geography:** Consistent across metros

The surviving entry-level postings are more genuinely junior: lower YOE requirements, more accessible language (FK 15.3 vs 16.5), and "years experience" bigram dropped from 5.6 to 2.1 per 1K characters (T13).

![YOE distribution changes](../assets/figures/T08/fig3_yoe_paradox.png)

### 4. Entry-level scope change (T11, corrected by T22)

The original T11 finding of +31pp management indicator at entry level was corrected by T22 to +4-10pp using validated patterns. However, AI requirements at entry level did surge genuinely (+23.6pp), and strategy/roadmap language showed a SWE-specific increase (+7pp DiD). The surviving junior roles demand more AI competency, but not dramatically more organizational scope.

## Critical caveat: Seniority operationalization discrepancy

The direction of the entry-level trend depends on which seniority column is used:

| Operationalization | 2024 entry share | 2026 entry share | Direction |
|-------------------|-----------------|-----------------|-----------|
| seniority_native (arshkon only) | 22.3% | 14.0% | **Decline** |
| seniority_final (arshkon only) | 20.4% | 14.5% | **Decline** |
| seniority_3level (T16 overlap panel) | 3.4% | 13.5% | **Increase** |

The T16 reversal is likely driven by different operationalization and panel composition (pooled 2024 includes asaniczka, which has zero native entry labels). The planned `seniority_llm` column will resolve this definitively. This is the highest-priority pending data improvement.

## Sensitivity

- Robust to company capping (sensitivity < 2pp at any threshold)
- Robust across 4 seniority operationalizations on direction (except seniority_3level in overlap panel)
- Robust to aggregator exclusion (direction holds; aggregators inflate absolute entry rates)
- Within-company decline (-11.8pp) controls for composition
- **Most specification-sensitive finding** -- requires formal robustness across all operationalizations

## Full analysis

- [T18: Cross-Occupation DiD](../reports/T18.md) -- SWE-specificity and DiD estimate
- [T03: Seniority Labels](../reports/T03.md) -- operationalization comparison
- [T06: Company Concentration](../reports/T06.md) -- within-company analysis
- [T08: Distribution Profiling](../reports/T08.md) -- YOE purification
- [T11: Requirements Complexity](../reports/T11.md) -- scope inflation (corrected by T22)
- [T22: Ghost Forensics](../reports/T22.md) -- management indicator correction
