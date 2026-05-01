# Verifications

Two adversarial verification passes audited the claim set independently, re-running the primary headlines against the raw data with minimal dependence on the original task scripts.

## V1 — Gate 2 adversarial verification

**Full report**: [V1_verification.md](tasks/V1_verification.md).

**Scope**: all 6 Wave 2 headlines (H1 through H6).

**Verdict**: all 6 verified within 5% of report magnitudes. H4 resolved Tension A (arshkon-only-~0 AND pooled-panel +3.5-6.4 pp are BOTH correct on different subsamples).

### The 4 V1 flags

1. **`mgmt_broad` precision 0.28** — all 4 broad tokens fail the 0.80 threshold. Retire the pattern.
2. **`mgmt_strict` precision 0.55** — sub-pattern `hire` at 0.07 ("contract-to-hire" / "how-we-hire" contamination); `performance_review` at 0.25 ("code review" / "peer review" contamination). V1 shipped `mgmt_strict_v1_rebuilt` at pilot 1.00 precision.
3. **`ai_broad` mcp_2024 at 0.15** — "MCP" = Microsoft Certified Professional in 2024. Direct within-2024 contamination. Drop mcp from ai_broad for 2024 baselines.
4. **T13 H3 classifier-sensitive** — J3 requirements-chars direction flips under an alternative simpler-regex classifier. Flag as classifier-uncertain.

### V1 artifact outputs

- `exploration/artifacts/shared/validated_mgmt_patterns.json` (7 primary patterns, later T22-extended).
- `exploration/tables/V1/` (sampled pattern matches, headline verifications, composite-score audits).

## V2 — Gate 3 adversarial verification

**Full report**: [V2_verification.md](tasks/V2_verification.md).

**Scope**: all 13 Wave 3 + Wave 3.5 headlines.

**Verdict**: all 13 verified on direction; 10 of 13 within 5% on magnitude. Two flags, both magnitude-level (direction robust in both cases).

### The 5 V2 flags (D1 through D5)

| ID | Report | Issue | Impact |
|---|---|---|---|
| D1 | Junior requirements shrank (T13) | Classifier-sensitive at aggregate (T33 confirmed flip) | **Demote to flagged qualified claim.** |
| D2 | Aggregate credential stack rose | Below within-2024 noise (SNR 0.59) | Cite only per-seniority. |
| D3 | MCP acceleration | `ai_broad` contamination in 2024 (Microsoft Certified Professional) | Use `ai_strict` only for MCP growth citations. |
| D4 | T16 + T23 pattern-provenance | Report text says v1_rebuilt but code uses top-level ai_strict | Explicit citation; direction unchanged; magnitude drops 10-15% under v1_rebuilt. |
| D5 | T31 pair-count | n=23 does not reproduce from documented methodology; V2 replicates n=37 or n=12 | **Range-report +10 to +13 pp.** Direction (pair > company) robust. |

### V2 Phase E — alt control-group definitions for SWE DiD

All 5 specifications produced SWE DiD within 0.5 pp of +13 to +14 pp:

| Specification | SWE DiD |
|---|---|
| V1-rebuilt pattern, default controls | +13.10 pp |
| Top-level pattern, default controls | +14.04 pp |
| Drop data/financial analysts | +13.18 pp |
| Drop nurse | +13.02 pp |
| Manual-work controls only | +13.44 pp |
| Drop title_lookup_llm SWE tier | +13.19 pp |

### V2 Phase B + C — cluster-0 robustness

- Phase B: 14 of 20 sampled cluster-0 titles are explicit AI/ML.
- Phase C: cluster-0 share robust across S1/S2/S3/S4 T30 panel variants at 10-14x rise each.

## What the verifications mean

- **6 of 6 Wave 2 headlines verified.**
- **13 of 13 Wave 3/3.5 headlines verified on direction; 10 of 13 within 5% on magnitude.**
- **Zero headlines reversed.** All flags are either pattern-provenance (fixable by citation) or magnitude-range (fixable by range-report).
- Pattern-validation protocol exposed a real 0.28-precision artifact in T11 — this is the methods contribution's most visible output.

## Reading order

1. Raw V1 report: [V1_verification.md](tasks/V1_verification.md).
2. Raw V2 report: [V2_verification.md](tasks/V2_verification.md).
3. How corrections were applied: [Measurement corrections](../findings/corrections.md).
