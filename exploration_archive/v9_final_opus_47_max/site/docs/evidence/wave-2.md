# Wave 2 — Open Structural Discovery

Let the data speak: what archetypes, boundaries, and content shifts fall out of unsupervised + supervised exploration?

| Task | Owner | Focus |
|---|---|---|
| [T08](tasks/T08.md) | Agent E | Distribution profiling + anomaly detection — 16 anomalies flagged |
| [T09](tasks/T09.md) | Agent F | **Posting archetype discovery — dominant axis is technology/role-type, NOT seniority (NMI 8.8x ratio)** |
| [T10](tasks/T10.md) | Agent G | Title taxonomy evolution — AI-token title share 9.6% → 23.5% |
| [T11](tasks/T11.md) | Agent G | **Requirements complexity + credential stacking — J3 breadth +1.58, S4 +2.61** |
| [T12](tasks/T12.md) | Agent H | Open-ended text evolution — period-dominant, not relabeling |
| [T13](tasks/T13.md) | Agent H | Linguistic + structural evolution — boilerplate-led length growth |
| [T14](tasks/T14.md) | Agent I | Technology ecosystem mapping — AI-family crystallizing |
| [T15](tasks/T15.md) | Agent I | **Semantic similarity landscape — junior-senior DIVERGING (0.946 → 0.863)** |

## Key Wave 2 findings

- Dominant clustering axis is **technology-domain / role-type**, not seniority (NMI ratio 8.8x).
- Junior and senior SWE postings are **diverging** (TF-IDF cosine 0.946 → 0.863), falsifying the relabeling prior.
- AI-mention acceleration is the cleanest cross-period signal (SNR 32.9).
- Length growth is boilerplate-led (benefits +89%, legal +80%, responsibilities +49%), not requirements-led.
- Scope inflation is universal; senior S4 +2.61 > junior J3 +1.58 (length-residualized).

## Wave 2 claims that required correction

- T11 "management declining" — corrected (V1 pattern at 0.28 precision; V1-rebuilt pattern shows flat).
- T13 "junior requirements shrank" — flagged classifier-sensitive (V1 alt-classifier gave +88%).
- T11 aggregate credential stack — below within-2024 noise; cite only per-seniority.
