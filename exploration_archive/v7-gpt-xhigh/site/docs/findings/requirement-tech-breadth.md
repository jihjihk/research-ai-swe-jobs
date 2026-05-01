# Requirements and technology breadth rose across seniority definitions.

<p class="lead">The strongest seniority-stratified content result is breadth expansion. It appears for junior and senior definitions, which weakens a simple downward-migration story.</p>

## Requirement breadth

In the LLM-cleaned SWE subset, T11 reports mean requirement breadth rising from 7.23 in pooled 2024 to 9.39 in scraped 2026 (+2.16). V1 independently verifies the result with slightly different regexes: 7.46 -> 9.67 (+2.20).

| Definition | 2024 | 2026 | Change |
|---|---:|---:|---:|
| J1: entry label | 6.26 | 8.00 | +1.74 |
| J2: entry/associate | 6.27 | 8.12 | +1.84 |
| J3: YOE <= 2 | 7.46 | 9.21 | +1.75 |
| J4: YOE <= 3 | 7.44 | 9.51 | +2.08 |
| S1: mid-senior/director | -- | -- | +2.40 |
| S4: YOE >= 5 | -- | -- | +2.61 |

Sources: [T11](../raw/reports/T11.md), [V1](../raw/reports/V1_verification.md).

<div class="figure-frame">
  <img src="../assets/figures/T11/junior_complexity_metric_changes.png" alt="Junior complexity metric changes from T11">
  <div class="figure-caption">T11 reports breadth growth under all J1-J4 definitions, while scope-density evidence is stronger for low-YOE definitions than for explicit entry labels.</div>
</div>

## Technology breadth

T14 reports that the shared SWE technology matrix expanded from 2024 to 2026. The biggest calibrated movers include CI/CD, Python, API design, observability, LLM, Kubernetes, AWS, generative AI, Docker, Terraform, and RAG.

| Technology | 2024 rate | 2026 rate | Change |
|---|---:|---:|---:|
| CI/CD | 15.4% | 33.6% | +18.2 pp |
| Python | 32.3% | 49.4% | +17.0 pp |
| API design | 13.0% | 27.4% | +14.4 pp |
| Observability | 1.9% | 13.9% | +12.0 pp |
| LLM | 1.0% | 13.0% | +12.0 pp |
| Generative AI | 0.9% | 7.5% | +6.7 pp |
| RAG | 0.1% | 5.2% | +5.1 pp |

Source: [T14](../raw/reports/T14.md).

<div class="figure-frame">
  <img src="../assets/figures/T14/tech_shift_heatmap.png" alt="Technology shift heatmap from T14">
  <div class="figure-caption">T14 separates calibrated risers from terms that rise but do not clearly exceed within-2024 source noise.</div>
</div>

## Text and boilerplate checks

T12's open-text comparisons identify 2026-heavy workflow/platform and exposure language: workflows, pipelines, familiarity, hands-on, tooling, observability, ownership, AI/ML, agent, LLM, and RAG. T13 shows that cleaned-text growth is not mostly benefits, about-company, legal, or compensation boilerplate.

Sources: [T12](../raw/reports/T12.md), [T13](../raw/reports/T13.md), [V1](../raw/reports/V1_verification.md).

## Boundaries

- Requirement breadth is regex-derived and restricted to LLM-cleaned rows.
- Tech breadth is binary mention evidence, not required proficiency evidence.
- Tech density per character does not generally rise; the safer claim is broader listed stacks in longer/richer postings.
- T22 failed, so requirement force remains unresolved.
