# SWE changed, but adjacent technical roles changed almost in parallel.

<p class="lead">The exploration no longer supports a SWE-only restructuring frame. The stronger cross-occupation result is that software-producing technical work absorbed AI/tool/platform requirements much faster than controls.</p>

## Evidence

| Metric | SWE 2024 -> 2026 | Adjacent 2024 -> 2026 | Control 2024 -> 2026 | SWE-control DiD |
|---|---:|---:|---:|---:|
| Broad AI, raw binary | 12.5% -> 39.7% (+27.2 pp) | 11.8% -> 38.8% (+27.0 pp) | 0.5% -> 3.1% (+2.6 pp) | +24.5 pp |
| AI-tool, raw binary | 2.2% -> 24.0% (+21.8 pp) | 2.1% -> 21.2% (+19.1 pp) | 0.01% -> 0.8% (+0.8 pp) | +21.0 pp |
| Bounded tech count | 5.00 -> 7.36 (+2.36) | 3.06 -> 5.37 (+2.31) | 0.22 -> 0.35 (+0.14) | +2.22 |
| Requirement breadth, LLM-labeled | 8.15 -> 10.71 (+2.56) | 6.39 -> 9.31 (+2.93) | 2.21 -> 2.92 (+0.70) | +1.86 |

Source: [T18](../raw/reports/T18.md).

<div class="figure-frame">
  <img src="../assets/figures/T18/did_core_metrics.png" alt="Difference-in-differences core metrics from T18">
  <div class="figure-caption">T18 finds a large SWE-control gap, but adjacent technical roles move close to SWE on the measured AI/tool and breadth surface.</div>
</div>

<div class="figure-frame">
  <img src="../assets/figures/T18/ai_gradient_by_occupation.png" alt="AI gradient by occupation from T18">
  <div class="figure-caption">The occupation gradient is strongest between software-producing technical work and controls, not between SWE and every adjacent role.</div>
</div>

## Interpretation

The right claim is SWE-amplified technical-work expansion. Data scientist/ML and data engineering roles are especially close to SWE on the 2026 measured surface, while controls show only modest background movement.

## Boundaries

T18 does not show generic occupation collapse. Its TF-IDF boundary check is mixed and does not support a clean SWE-adjacent semantic convergence claim. Requirement-breadth metrics are LLM-labeled subset claims, and raw AI mentions are not force-coded.
