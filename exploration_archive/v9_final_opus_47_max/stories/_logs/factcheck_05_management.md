# Fact-check: Piece 05 "A decline that wasn't"

**Fact-checker:** Claude Opus 4.7 (sub-agent), 2026-04-21
**Sources consulted:** `exploration/reports/V1_verification.md` §2; `exploration/reports/T21.md`; `exploration/artifacts/shared/validated_mgmt_patterns.json`.

## Claims vs. sources

### Claim 1 — mgmt_broad precision = 0.28; all 4 broad sub-patterns below 0.80
- **V1_verification.md §2:** mgmt_broad precision = **0.28**; listed failing broad tokens: lead 0.12, team 0.08, stakeholder 0.18, coordinate 0.28, manage 0.22 (five, not four).
- **validated_mgmt_patterns.json:** mgmt_broad precision 0.28; broad sub-patterns all below 0.80 (lead 0.12, team 0.08, stakeholder 0.18, coordinate 0.28, manage 0.22).
- **Verdict:** MATCHES on 0.28; the piece says "4 broad sub-patterns" but there are actually 5 broad extensions (lead/team/stakeholder/coordinate/manage). Minor count error — claim understates, does not overstate.

### Claim 2 — mgmt_strict precision = 0.55; hire 0.07; performance review 0.25
- **V1_verification.md §2 table:** mgmt_strict 0.55; hire 0.07; performance_review 0.25. Exact match.
- **validated_mgmt_patterns.json:** precision 0.55; hire 0.07; performance_review 0.25. Exact match.
- **Verdict:** MATCHES exactly.

### Claim 3 — Senior mgmt density under v1_rebuilt: 0.039 → 0.038 (flat)
- **T21.md §2 (density_summary_by_period_seniority):** mid-senior mgmt_rebuilt 0.039 (2024) → 0.038 (2026), Δ −0.001, within-2024 SNR 0.1.
- **T21 headline:** "mid-senior mgmt density is flat (0.039 → 0.038, Δ≈0)".
- **Verdict:** MATCHES exactly.

## Additional checks

- **semantic_precision_measured on mgmt_strict_v1_rebuilt:** `true` (JSON line 176).
- **Rebuilt precision:** JSON records 0.98 (V1 in-file, line 178) and T22 re-validation 0.98 (line 323). Piece claims 0.98 — matches. By-period: 2024 = 1.00, 2026 = 0.96. Both ≥ 0.95.
- **Sub-pattern spot-check:** mentoring junior 1.00 (n=21); mentoring others 1.00 (n=10); coach junior 1.00 (n=4); headcount 0.5 (n=2, noted weakness but small n).

## Final verdict

**MATCHES** — all three headline numbers verify against source files. One qualification: the piece says "all 4 broad sub-patterns below 0.80"; V1 lists 5 broad extensions, all failing. Recommend editing "4" to "5" or rephrasing to "every broad extension". Piece 05's quantitative claims are otherwise exact.
