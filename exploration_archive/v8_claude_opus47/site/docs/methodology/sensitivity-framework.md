# Sensitivity framework

The exploration built a measurement framework alongside the substantive findings. Three elements: the T30 seniority panel, the 9 sensitivity dimensions, and the composite-score correlation protocol.

## T30 seniority panel

Every seniority-stratified headline reports a four-row ablation. The panel is stored at `exploration/artifacts/shared/seniority_definition_panel.csv`.

### Junior definitions

| Def | Rule | Source dependency | Report as |
|---|---|---|---|
| J1 | `seniority_native = 'entry'` | arshkon-only (asaniczka has 0) | label-strict, diagnostic |
| J2 | `seniority_final IN ('entry', 'associate')` | Combined rule + LLM | primary |
| J3 | `yoe_extracted ≤ 2` | Label-independent | YOE-based, stress test |
| J4 | `yoe_extracted ≤ 3` | Label-independent | generous YOE |

### Senior definitions

| Def | Rule | Report as |
|---|---|---|
| S1 | `seniority_final IN ('mid-senior', 'director')` | primary |
| S2 | `seniority_final = 'director'` | sparse-cell diagnostic |
| S3 | title contains `staff\|principal\|lead\|architect` | title-keyword |
| S4 | `yoe_extracted ≥ 5` | label-independent |

### Why four rows

The exploration found the junior-share direction **flips between J1 and J3 under the same baseline**, and **flips between arshkon-only and pooled-2024 under the same definition**. Any paper headline based on a single definition would be a measurement-regime artifact. Reporting all four rows × two baselines forces the reader to see the direction's baseline-contingency directly.

## Nine sensitivity dimensions

Applied to every Wave 2+ headline as pre-committed at Gate 0:

| # | Dimension | Test |
|---:|---|---|
| 1 | Aggregator exclusion | Re-run with `is_aggregator = false` |
| 2 | Entry-specialist exclusion | Re-run without 240 T06-flagged specialists |
| 3 | Cap-50 per company per period | No single company dominates aggregate |
| 4 | Arshkon-only vs pooled-2024 baseline | Direction should not depend on pooling choice |
| 5 | Labeled-only LLM-frame vs full corpus | Detects LLM-frame selection artifacts |
| 6 | Length-residualization | On any composite with component r > 0.3 against length |
| 7 | Strict vs broad regex pattern | Strict is primary; broad as sensitivity |
| 8 | T29 authorship-score bottom-40% | Recruiter-LLM mediation retention |
| 9 | DiD vs control occupations | SWE-specificity check |

A finding that does not survive these is not yet a finding.

## Signal-to-noise rule

SNR ≡ (cross-period effect) / (within-2024 cross-source gap).

- **SNR ≥ 2** — material finding.
- **SNR 1-2** — marginal; require additional sensitivities.
- **SNR < 1** — within-2024 noise exceeds cross-period effect; not a headline.

T05's SNR table is the single most consequential number in Wave 1. Junior metrics SNR: J1 = 0.19, J2 = 0.24, J3 = 0.43 — all well below 2. AI-strict: SNR 35.4. That gap is why AI-rewriting is the paper's lead.

## Semantic precision protocol

For every new regex pattern:

1. Sample 50 matches stratified 25/25 by period (arshkon/asaniczka vs scraped).
2. Manually label: does the match communicate what the pattern claims?
3. Pattern passes at **80% precision floor**.
4. Failures dropped; compound patterns re-run without the failing sub-term.

V1 applied this to management and AI patterns. Failures:

| Term | Precision in SWE JDs | Why |
|---|---:|---|
| bare `manage` | 14 % | "manage data", "manage state", "manage systems" dominates |
| `stakeholder` | 42 % | "communicate with stakeholders" is boilerplate |
| `agent` | 44 % | "agent-based", "sales agent", "recruitment agent" contaminate |
| `mcp` | 57 % | "MCP certification" and other non-protocol uses |
| `team_building` | 10 % | rarely appears; when it does, it's benefits text |

Refined strict management pattern: `mentor|coach|hire|headcount|performance_review`. All sub-terms pass 80%; `mentor` alone is 100% on a 20-sample.

## Length residualization protocol

V1 found composite-score components correlate with description length:

| Component | r vs desc_cleaned_length |
|---|---:|
| Tech count | 0.48 |
| Management-strict | 0.35 |
| Org-scope count | 0.40 |
| Soft-skill count | 0.36 |
| AI-strict binary | 0.11 (near-zero — robust) |

Global OLS fit on `requirement_breadth`: `a=6.498, b=0.00182`. Residualized breadth is the primary metric for all T16 within-company claims; raw breadth is sensitivity only.

The 71%/29% content-vs-length decomposition on requirement_breadth rise: 71% of the +39% raw rise is real content; 29% is length-driven.

## Composite-score correlation check

For any multi-component composite, report the correlation matrix among components and between each component and description length. If any component-pair r > 0.7 or any component-length r > 0.3, residualize.

## DiD against control

For any SWE-specific claim, test against:

1. `is_swe_adjacent` (ML-eng, data-sci, data-eng, security-eng, solutions-arch) — technical roles with some code.
2. `is_control` (nurses, accountants, marketers) — non-technical occupations.

If the SWE effect is not distinguishable from the same effect in adjacent + control, the claim is field-wide not SWE-specific. Bootstrap 400 replicates, 95% CI. T18 is the template.

## Authorship-score mediation test

T29 computes per-posting authorship-score (LLM-drafted likelihood) from within-period feature distributions. Any AI-content claim must report the headline Δ on:

- Full corpus (primary)
- Low-40% within-period subset (sensitivity)

AI-strict retains 75-77% across score specifications — robust. Mentor/breadth retain 0-30% depending on score specification — method-sensitive. Findings framed accordingly.

## Tools and artifacts

- `exploration/artifacts/shared/seniority_definition_panel.csv` — 44 rows (junior J1-J6 × senior S1-S5 × sources).
- `exploration/artifacts/shared/calibration_table.csv` — AI-mention + breadth benchmarks by source.
- `exploration/artifacts/shared/validated_mgmt_patterns.json` — 8 patterns, all ≥ 80% precision.
- `exploration/artifacts/shared/swe_tech_matrix.parquet` — 107 boolean tech cols (Stage 9 LLM-derived).
</content>
</invoke>