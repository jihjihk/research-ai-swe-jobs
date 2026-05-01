# Verification

Two adversarial verification rounds. Each agent was given the gate memo and asked to re-derive the strongest claims independently, with permission to disagree.

## V1 — Gate 2 adversarial re-derivation

Read: [V1_verification.md](verification/V1_verification.md).

Six Gate 2 claims re-derived. Five verified; one corrected.

| Claim | V1 verdict | Note |
|---|---|---|
| AI-mention pattern semantic precision | Verified (refined) | Dropped `agent` (66%), `mcp` (57%) |
| Requirements-section shrink | Verified | SWE -19% holds |
| `requirement_breadth` length correlation | Corrected | Decomposed to 71% content / 29% length |
| NMI domain/period/seniority | Verified | 8.6× domain/period on full corpus |
| Within-LLM-frame J2 flip | Verified | 2-3× labeled-vs-not-labeled gap confirmed |
| Relabeling cosines (seniors changed more) | Verified (magnitude corrected) | "180× period/seniority" corrected to ~1.2× centroid-pairwise |

## V2 — Gate 3 adversarial re-derivation

Read: [V2_verification.md](verification/V2_verification.md).

Eight Gate 3 claims re-derived. Five verified, two flagged for method-sensitivity, one corrected.

| Claim | V2 verdict | Note |
|---|---|---|
| T16 within-co 102% on AI-strict | Verified exact | |
| T18 DiD CI bounds | Verified | Bootstrap clear of zero |
| T20 yoe-excluded panel +0.134 AUC | Verified exact | |
| T23 RQ3 direction | Verified | All benchmarks |
| T28 AI/ML 81% new-entrant | Verified exact | |
| T28 ≥+10pp attribution | **Corrected** | Is BROAD, not STRICT (Wave 3 summary conflated) |
| T29 mentor retention 72% | Flagged | Method-sensitive (105% under 3-feature score) |
| T29 breadth retention 71% | Flagged | Method-sensitive |

V2's primary contribution: establishing that AI-strict retention is robust across score specifications, but mentor/breadth are not. Paper should cite mentor/breadth with explicit "0-30% mediated with method uncertainty."
</content>
</invoke>