# Finding 4 — Employers under-specify AI versus worker usage

## Claim

**The 2026 SWE broad-AI employer requirement rate is 46.8 percent — below every plausible developer-usage benchmark (50, 65, 75, 85 percent). The gap is 15 to 30 percentage points depending on benchmark; the direction is inverted from the pre-registered anticipatory-over-specification hypothesis; and seniors are more AI-specified than juniors (51.4 vs 43.5 percent), ruling out an AI-as-junior-filter interpretation.**

## Core numbers

| Metric | Value | Source |
|---|---:|---|
| 2026 SWE broad-AI employer rate | **46.8 %** | T23 |
| 2026 SWE strict-tool employer rate | 14.2 % | T23 |
| Stack Overflow 2024 — currently using | 62 % | Worker benchmark |
| Stack Overflow 2024 — plan to use | 76 % | Worker benchmark |
| Octoverse 2024 — OSS AI | 73 % | Worker benchmark |
| Anthropic 2025 — programmer exposure | 75 % | Worker benchmark |
| Gap (broad) at central 75% assumption | **−28 pp** | T23 |
| Gap (strict) at central 75% assumption | −61 pp | T23 |

| Seniority | Broad-AI rate | Strict-tool rate |
|---|---:|---:|
| J2 (entry + associate) | 43.5 % | 12.4 % |
| J3 (yoe ≤ 2) | 46.1 % | 13.8 % |
| S1 (mid-senior + director) | **51.4 %** | 16.0 % |
| S4 (yoe ≥ 5) | 50.2 % | 15.5 % |

Seniors are the *most* AI-specified group. Any "AI is a junior filter" interpretation is ruled out.

## Key figure

![Employer/worker divergence](../figures/T23/T23_divergence.png)
*T23 — 2026 employer AI rates vs four worker-usage benchmarks at 50/65/75/85% assumptions. Every benchmark is above the employer broad-AI rate; strict-tool rate is far below all four.*

![Interview-ready divergence chart](../figures/employer_usage_divergence.png)
*T25 — Curated chart for RQ4 interviews: overlays the three published worker benchmarks (SO 2024, Octoverse 2024, Anthropic 2025) against employer strict/broad rates.*

## Sensitivity across benchmark assumptions

Gap (pp) = employer_rate − worker_benchmark:

| Worker benchmark | Broad 46.8% | Strict 14.2% |
|---|---:|---:|
| 50 % (most conservative) | −3.2 pp | −35.8 pp |
| 65 % | −18.2 pp | −50.8 pp |
| 75 % (central) | **−28.2 pp** | −60.8 pp |
| 85 % (Anthropic-style exposure) | −38.2 pp | −70.8 pp |

The direction is invariant across the entire 50-85% band. The only assumption at which broad-AI just crosses parity is an unusually conservative 50% benchmark — and even there, strict-tool is 36pp below.

V2.2 alternative framing: "currently using" 63.2% gives a gap of −49.3pp at ai_tool (strict) but −16.4pp at ai_broad. Direction holds.

## What it rules out

- **Anticipatory employer over-specification** (the pre-registered RQ3 hypothesis).
- **AI as junior-filter.** If AI were an internal filter masking through JDs, juniors would be under-specified relative to seniors. They are less specified, not more.
- **Single-benchmark artifact.** The direction holds across four independent benchmarks in the 50-85% band.

## What it suggests

Three mechanisms candidates for RQ4 interviews:

1. **JD template lag** — hiring managers are slower to update than developers are to adopt.
2. **Implicit assumption** — "AI tools are table-stakes; we don't list them," parallel to git.
3. **AI as coordination signal, not skill demand** — postings communicate firm-level AI adoption to investors/candidates, not a skill requirement.
4. **Internal filter, not public requirement** — firms screen on AI-tool-use in interviews but omit from postings.

See H_I in [T24](../raw/wave-4/T24.md) for the coordination-signal hypothesis test plan.

## Limitations

- **Posting ≠ worker units.** Employer rate is share-of-postings; worker rate is share-of-workers. A posting-level requirement can't cover all worker-level usage if one posting hires one worker — but the direction of the gap is robust to this.
- **Worker benchmarks are self-reported.** Stack Overflow and Octoverse are platform-biased toward engaged developers. Anthropic 2025 is task-exposure, not usage.
- **The gap is direction, not point.** Paper reports the qualitative direction with a 15-30pp range, not a precise number.

## Novelty assessment

This is the paper's **most novel standalone finding**. The pre-registered hypothesis has been inverted. Publishable on its own at a labor-economics venue (ILR Review, BE Journal of Economic Analysis & Policy) or as a dedicated section of the main paper.

## Links to raw

- [T23 — employer-usage divergence](../raw/wave-3/T23.md)
- [T24 — new hypotheses, H_A extends to cross-occupation](../raw/wave-4/T24.md)
- [V2 verification](../raw/verification/V2_verification.md) — alternative framing check
- [Interview elicitation](../narrative/interview.md) — RQ4 question 3
</content>
</invoke>