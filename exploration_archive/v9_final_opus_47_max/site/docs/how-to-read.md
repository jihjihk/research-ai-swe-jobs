# How to read this site

The three top tabs answer three different reader questions.

## "Give me the overview"

The **Overview** tab has:

- The 23-slide deck (linked from the index and embedded at [Slide deck](presentation.md)). Each slide title is a complete-sentence claim. If you only read titles, you still get the whole story.
- A 500-word [Executive summary](summary.md).

## "Tell me more about claim X"

The **Findings** tab has one page per Tier A lead finding, plus a page on rejections and a page on measurement corrections. Every claim page has the same shape:

1. A one-sentence claim.
2. The key figure or table.
3. Evidence with task citations.
4. A sensitivity verdict.
5. A link to the raw task report in the Audit-trail tab.

Start with A1 (cross-occupation divergence) if you want the biggest-magnitude result. Start with A2 (within-firm rewriting) if you care about firm-level dynamics. Start with A3 (seniority sharpening) if you want the cleanest counter-intuitive finding.

## "Should I trust this?"

The **Methodology** tab covers:

- What data we used (three LinkedIn panels, date windows, filter logic).
- How we preprocessed — pipeline stages, the LLM extraction and classification prompts, and coverage caveats.
- The sensitivity framework (9 dimensions, within-2024 noise floors, pattern validation).
- What the paper can and cannot claim.

## "Show me the raw work"

The **Audit trail** tab contains everything — 34 task reports, two adversarial verifications, four gate memos, the full synthesis. Organized by wave. If you want to audit one specific claim, this is where you click through from the Findings-tab link.

## Conventions used throughout

- **Task IDs** (T01 through T38) are hyperlinks into the Audit-trail tab.
- **Pattern names** like `ai_strict_v1_rebuilt` are the validated regexes documented in `validated_mgmt_patterns.json`.
- **Sample codes** like **J3** (YOE LLM <= 2), **S4** (YOE LLM >= 5), **J1/J2/J4/J5/J6/S1/S2/S3/S5** are from the T30 seniority panel.
- **SNR** numbers are cross-period-effect divided by within-2024-effect (arshkon vs asaniczka on the same metric). SNR > 2 is above-noise; 1 to 2 is near-noise; < 1 is below-noise.
- **pp** is percentage points.
- **DiD** is difference-in-differences.
