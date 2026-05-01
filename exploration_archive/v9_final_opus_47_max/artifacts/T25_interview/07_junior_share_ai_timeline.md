# Artifact 7 — Junior-share trend and AI-model release timeline

**Source:** T08 seniority share + T19 rate-of-change + temporal context. `exploration/figures/T08/fig3_seniority_final.png`, `exploration/figures/T19/fig1_timeline.png`.

This artifact situates posting content changes against the macro AI-model release timeline between the 2024 baseline and 2026 scraped window.

---

## Data snapshot facts (T19)

**Three analytical observation points:**

| Source | Midpoint | n (SWE LinkedIn English date-flag-ok) |
|---|---|---|
| asaniczka | 2024-01-14 | 18,129 |
| arshkon | 2024-04-12 | 4,691 |
| scraped (pooled 2026-03 + 2026-04) | 2026-04-05 | 45,317 |

Arshkon → scraped midpoint distance: **~791 days (2.17 years)**.

## Key AI model releases between the observation points

- **2024-01:** asaniczka snapshot. GPT-4 (Mar 2023) and GPT-4 Turbo (Nov 2023) available. Claude 2 current; Claude 3 not yet released. Copilot available but GA <18 months.
- **2024-03:** Claude 3 Opus released.
- **2024-04:** arshkon snapshot. Claude 3 Opus + GPT-4 Turbo available. GPT-4o and Claude 3.5 Sonnet weeks away.
- **2024-05:** GPT-4o released (multimodal).
- **2024-06:** Claude 3.5 Sonnet released (coding-capable frontier).
- **2024-09:** OpenAI o1 (reasoning) released.
- **2024-12:** DeepSeek-V3 released.
- **2025-02:** GPT-4.5 released.
- **2025-03:** Claude 3.6 Sonnet (Claude 3.7 Sonnet announced 2025-02 equivalently).
- **2025-09:** Claude 4 Opus released.
- **2026-03:** Gemini 2.5 Pro released.
- **2026-04:** scraped midpoint. Present-vintage frontier (Claude 4, GPT-4.5, Gemini 2.5).

**~8 major frontier model releases between arshkon and scraped.** No single release can be attributed as the driver; the window encompasses the entire "agent-era" arrival and broad enterprise adoption of frontier coding assistants.

---

## Junior-share and AI-rate trajectory

| Metric | 2024 pooled | 2026 scraped | Δ | Within-2024 noise | Acceleration ratio (T19) |
|---|---|---|---|---|---|
| J3 share (yoe≤2) | 9.15% | 14.19% | **+5.04 pp** | +4.75 pp (arshkon vs asaniczka) | 0.12 (below 1 — within-2024 noise dominates annualized rate) |
| S4 share (yoe≥5) | 70.77% | 63.18% | **−7.59 pp** | 7.09 pp | 0.12 (below 1 — dominated by short 2024 window) |
| ai_strict share | 1.47% | 14.93% | **+13.46 pp** | 0.5 pp (near zero) | **2.80** (clean rate signal) |
| scope share | 43.07% | 63.77% | +20.70 pp | ~6.2% | 1.54 (borderline above noise) |

## Why the AI signal is the cleanest rate-signal and seniority is not

T19 formalized: the within-2024 period has only ~89 days (asaniczka to arshkon midpoint). Any tiny within-2024 difference translates into a large annualized rate, collapsing seniority SNR below 1. For AI and scope, the within-2024 noise is tiny in absolute terms, so SNR >1 is achievable.

**Practical upshot:** seniority claims should be cited as raw pp deltas (T30 panel primaries J3 +5.04 pp, S4 −7.59 pp pooled / −1.78 pp arshkon-only); annualized rates are a poor frame. AI claims can use either framing — the signal is above noise on both.

---

## Interview questions

### Temporal attribution

1. **"Between early 2024 and early 2026, there were 8+ major frontier-AI model releases. Do you associate any specific release (GPT-4o, Claude 3.5 Sonnet, Claude 4 Opus, Gemini 2.5) with a visible shift in how your firm writes SWE postings?"**

2. **"Our data is three snapshots, not a continuous time-series. We cannot directly attribute content shifts to specific model releases. What would you recommend we look for in an analysis-phase continuous-time-series panel? What events should we treat as candidate 'treatments' for an event-study?"**

### Junior-rise mechanism

3. **"The data shows J3 entry-share rose by 5 pp between 2024 and 2026 — but the rise is almost entirely in the EXIT of senior-heavy 2024 firms, not in new junior postings at returning firms. 4,395 firms left the panel; they were senior-heavy. Returning firms sharpened their senior-vs-junior distinction without adding juniors. Does this match the hiring mix shifts you observed at your firm or in your sector?"**

4. **"2026 is a hiring trough for Information (JOLTS openings at 0.71× 2023 average). Yet SWE content shifted toward AI and scope broadening. The data shows hiring-volume direction is NOT correlated with content-shift direction (T38). Why might employers invest in content-rewriting during a hiring slowdown?"**

### AI-adoption chronology

5. **"The ai_strict posting rate grew from 1.03% (2024) to 10.6% (2026) — a 10× rise. This is one of the cleanest rate signals in the entire dataset (acceleration ratio 2.80, SNR 32.9). Does this match your internal-hiring reality? Was the 2024 baseline of 1% 'too low' even at the time, or is 10% a genuine new formalization level?"**

6. **"GitHub Copilot has been in the market since 2022 and reached ~33% regular-use by 2024. The employer posting rate for 'copilot' went from 0.004% (2024) to 0.10% (2026). Why didn't 2024 postings formalize Copilot? What changed in 2026 that added even this small level of formalization?"**

---

## Limitations for interview framing

- **We cannot attribute specific posting shifts to specific model releases.** The 791-day window encompasses 8 major releases. Interview informants may offer attributions, but these are informant perceptions, not validated quantitative claims.
- **The 2024 asaniczka and arshkon snapshots are NOT a "within-2024 trend" — they are a cross-instrument noise floor.** Asking interviewees to reason about "trend within 2024" relies on external evidence (e.g., their own firm's hiring data), not this exploration's data.
- **Posting age is nearly unobservable** (0.9% of scraped rows have `posting_age_days`; all are 1 day). Posting lifecycle analysis is out of scope.

## Chart annotation suggestions for paper figure

If a junior-share + AI-timeline figure is produced for the paper:

- X-axis: time, with arshkon / asaniczka / scraped markers at their respective midpoints.
- Y-axis 1: junior share (J3, yoe≤2), senior share (S4, yoe≥5) — left axis.
- Y-axis 2: ai_strict rate — right axis.
- Annotations: GPT-4o (2024-05), Claude 3.5 Sonnet (2024-06), Claude 4 Opus (2025-09), Gemini 2.5 Pro (2026-03).
- Cite T19 acceleration ratio 2.80 for AI; note that seniority changes are reported as raw pp deltas per T19 methodological caveat.
