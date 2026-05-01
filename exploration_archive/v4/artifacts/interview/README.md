# T25 — Interview elicitation artifacts

Seven visual artifacts for qualitative interviews with senior engineers,
junior engineers, and hiring-side actors (HR, recruiters, engineering
managers). Each artifact is keyed to one question, anchored in a specific
Wave 2/3 finding.

All figures are 150dpi PNG. Companies anonymized where postings are
individually identifiable. Designed to be shown on screen or printed
on letter paper.

## How to use

- Hand the figure to the interviewee and let them look for 30-60 seconds.
- Ask the prompt printed in the title.
- Follow-up prompts (below) probe the specific finding without leading.
- Do NOT start with the "correct" answer — let the interviewee reframe
  what they're seeing.

---

## Artifact 1 — Inflated entry-level job descriptions

File: `artifact_1_inflated_junior_jds.png`
Source: T22 `top20_ghost_entry_combined.csv` (kitchen-sink score +
aspiration ratio); companies anonymized.

Four panels show real 2026 entry-level or new-grad SWE postings that
stack 15-25 technology mentions, multiple scope responsibilities,
and (in one case) a TS/SCI clearance requirement on a "Skill Level 0"
title. Each panel includes a 700-character excerpt of the actual
posting text and a short annotation describing why the posting is
striking.

Interview prompt: *"Is this posting a real role or aspirational
signaling?"*

Follow-ups:
- "Would you actually expect a new graduate to do all of this?"
- "Which of these listed skills do you think are load-bearing?"
- "Have you ever applied to or posted a role like this?"

Target audiences: junior engineers (candidate experience), hiring
managers (what they're actually screening on), HR (why the list exists).

---

## Artifact 2 — Same-company JD pairs, 2024 vs 2026

File: `artifact_2_paired_company_jds.png`
Source: T16 `overlap_per_company_changes.csv` — four companies with the
largest Δ AI rate and Δ description length across the overlap panel.

Four 2-column rows; each row is one (anonymized) company with its 2024
posting on the left and its 2026 posting on the right. Descriptions
are excerpted to ~620 chars per panel with char-count and title shown.
Annotations identify the specific contrast (AI-rate, length, scope).

Interview prompt: *"What changed in your hiring philosophy between
these two postings?"*

Follow-ups:
- "Is the 2026 role actually a different job, or the same job written
  differently?"
- "Which posting would you find easier to screen candidates for?"
- "Did your team rewrite your job-description template between 2024
  and 2026?"

Target audiences: hiring managers, recruiters, senior engineers who
have been at the same employer across the period.

---

## Artifact 3 — Entry-share trend under multiple operationalizations

File: `artifact_3_entry_share_operationalizations.png`
Source: T08 `14_entry_share_ablation.csv` and `15_yoe_proxy_entry_share.csv`.

Single panel showing three lines (native label; combined rule+LLM
column; YOE≤2 proxy) connecting 2024 and 2026 entry-share values.
The visual story is the directional disagreement: native label
DECLINES ~22→14% (but is contaminated), combined column RISES
~3→9%, YOE proxy RISES ~10→17%. Major LLM model-release dates are
shown as vertical annotations. Inline summary box explains the
measurement trap.

Interview prompt: *"Did your team change its junior hiring around
any specific model release?"*

Follow-ups:
- "Was there a moment when your team pulled back on new-grad hiring?"
- "Was there a moment when your team started requiring AI
  experience in JDs?"
- "Do you notice the rise, the decline, or neither in your own
  pipeline?"

Target audiences: hiring managers, engineering directors, recruiters
who track pipeline metrics.

---

## Artifact 4 — Senior 3-axis archetype shift

File: `artifact_4_senior_archetype_shift.png`
Source: T21 `senior_density_by_period_raw.csv` and
`senior_kmeans_shares_by_period.csv`.

Two panels. Left: 2024 vs 2026 any-share bars for four senior-tier
language profiles (people-management, mentoring, tech-orchestration,
strategic scope). Right: stacked bars showing the 5-cluster k-means
reorganization — People-Manager cluster 3.7→1.1% (−70%),
TechOrch-heavy cluster 5.0→7.6% (+52%), Mentor-heavy 8.6→10.8%
(+26%).

Interview prompt: *"Is the IC + mentoring shift real in your
experience, or is it just JD framing?"*

Follow-ups:
- "Did your own responsibilities shift away from people management
  between 2024 and 2026?"
- "Are you mentoring more junior engineers now than you were two
  years ago?"
- "Do you recognize the new tech-orchestration vocabulary
  (agentic, guardrails, prompt engineering) as describing what
  you actually do?"

Target audiences: mid-senior and senior ICs, engineering managers
who have been at the senior level across the period.

---

## Artifact 5 — AI vocabulary that emerged from zero

File: `artifact_5_ai_vocabulary_emergence.png`
Source: T21 `senior_term_level_counts_capped20.csv`, T22
`pattern_validation_counts.csv`, T23 `ai_requirement_rates`.

Horizontal bar chart for 11 AI terms with 2024 rate (near zero)
and 2026 rate (material): `agentic`, `rag`, `ai agent / multi-agent`,
`prompt engineering`, `guardrails`, `langchain`, `cursor`, `claude`,
`copilot`, `llm / llms`, `model context protocol`. Each row shows
the 2024→2026 multiplier in the right margin.

Interview prompt: *"Are you and your team actually using these
tools, or is this aspirational?"*

Follow-ups:
- "Which of these terms do you use in day-to-day work?"
- "Which of these terms do you think your employer uses because
  they feel like they have to?"
- "Did you learn about 'agentic' from a job posting, or did the
  posting learn it from you?"

Target audiences: all — the question splits cleanly by
practitioner vs author-of-JD.

---

## Artifact 6 — Posting-usage divergence (RQ3 inverted)

File: `artifact_6_posting_usage_divergence.png`
Source: T23 `ai_requirement_rates_direct_only.csv` (employer) +
Stack Overflow Developer Survey 2024/2025 (worker).

Two panels. Left: any-AI employer requirement rate (11%→53%) vs
worker AI usage rate (62%→80%), with the −50pp→−27pp gap
annotated. Right: agentic/AI-agent specifically (0→8% employer
vs 0→24% worker). The central story is "workers were ahead,
stayed ahead; gap is closing not opening".

Interview prompt: *"Did you start using AI tools before your
employer asked you to?"*

Follow-ups:
- "When did you first use Copilot / Cursor / Claude at work?"
- "When did your JDs first mention those tools?"
- "Are you using AI agents today? Has your employer asked about it?"

Target audiences: working engineers (self-report their own
adoption); hiring managers (contrast to their JD language).

---

## Artifact 7 — Defense contractors over-represented in entry posting

File: `artifact_7_defense_contractor_entry.png`
Source: T16 `top20_entry_posters_scraped.csv` and
`entry_poster_industries.csv`.

Two panels. Left: ranked horizontal bars of the top 20 scraped
2026 entry posters, with defense/aerospace contractors highlighted
in red (SpaceX, Booz Allen, Northrop Grumman, Leidos, Peraton,
Raytheon). Right: annotation box summarizing the finding —
90.4% of scraped companies (≥5 SWE) post zero entry rows under
the combined column; defense contractors make up ~25% of the
top-20 entry volume.

Interview prompt: *"Why does your organization (or your employer)
run a new-grad program when most don't?"*

Follow-ups:
- "Is the clearance-track pipeline the reason you can support
  a new-grad program?"
- "Did your company's entry posting volume change between 2024
  and 2026, or is it structural?"
- "Is there a reason 'tech' employers are less active in
  entry-level than defense contractors?"

Target audiences: engineering managers at defense/aerospace
employers, new-grad recruiters, candidates who landed at
defense contractors vs tech companies.

---

## Figure-generation

Script: `exploration/scripts/T25_interview_artifacts.py`
Run with: `./.venv/bin/python exploration/scripts/T25_interview_artifacts.py`

All figures 150 dpi PNG, white background, matplotlib/numpy only.
Data tables referenced by each figure are listed in the source
note at the bottom of the figure and in the section above.
