# Interview Design: Mechanisms of SWE Posting Restructuring

This is the **canonical qualitative design doc**. It defines the interview cohorts, protocol logic, elicitation materials, and analytic role of the qualitative study within the mixed-methods paper.

## Purpose

This qualitative study is the mechanism arm of the project. The quantitative analysis establishes **what changed in SWE postings**. The interviews explain:

- who changed job descriptions
- why the bar moved
- whether listed requirements are real, screened, or aspirational
- how AI is changing actual work
- whether employers are reacting to current AI capability or to expectations about future capability

The interview study is therefore not a generic supplement. It is how we validate the most novel construct in the paper: **anticipatory restructuring**.

## Qualitative Orientation

The study uses explanatory sequential mixed methods with a reflexive thematic-analysis approach.

Two goals matter:

1. Mechanism identification
2. Capturing the lived experience and professional meaning of the shift

That combination matters because posting text alone cannot tell us whether a requirement is:

- operationally real
- template inflation
- recruiter overreach
- manager signaling
- defensive hiring in a weaker market

## What the Interviews Add

| Gap in posting data | What interviews reveal | Why it matters |
|---|---|---|
| A JD changed | Who changed it and who approved it | Core mechanism evidence |
| A requirement appears in text | Whether it is truly screened or mostly ornamental | Validates ghost requirements |
| Senior postings use less management language | Whether senior work really shifted away from mentoring and people management | Validates senior archetype shift |
| Junior postings look inflated | Whether the hiring bar actually moved or only the written description did | Distinguishes real restructuring from template inflation |
| AI terms rise in postings | Whether firms are reacting to actual workflows or to AI hype / strategic signaling | Validates anticipatory restructuring |
| Breaks appear in time series | Whether practitioners remember a sudden shift, a gradual shift, or no shift | Context for break analysis |

## Recommended Cohorts

The prior two-cohort design was good, but the stronger design adds a third cohort that sits on the hiring side.

### Cohort A: Senior SWEs and tech leads

Target `n=8`

Variation to seek:

- IC vs. people-manager track
- hiring involvement
- heavy AI users vs. skeptics
- big tech vs. startup vs. mid-size company
- 5-8 years vs. 10+ years of experience

### Cohort B: Junior engineers and CS students

Target `n=8`

Variation to seek:

- current students vs. 0-2 YoE engineers
- actively searching vs. recently hired vs. struggling to break in
- top programs vs. state schools vs. bootcamp backgrounds
- heavy AI users vs. light users
- multiple metros

### Cohort C: Hiring-side actors

Target `n=8`

Include some mix of:

- engineering managers
- recruiters
- talent leads
- founders at smaller firms

This cohort is essential for the strongest version of the paper because it directly addresses:

- who authored the JD
- who raised the bar
- whether AI requirements are screened
- whether legal / HR templates inflated the posting

### Sample size logic

Preferred target is `24` total interviews. If recruitment is difficult, a minimum viable design is `18` with `6` per cohort.

## Recruitment Strategy

Likely recruitment channels:

- personal and professional network
- alumni groups
- startup founders and EMs in the existing network
- recruiter referrals
- snowball sampling after each interview

Limitations should be explicit:

- non-random sample
- likely overrepresentation of tech-aware participants
- likely overrepresentation of network-adjacent demographics

Mitigations:

- maximum-variation screening
- transparent sample table in the paper
- explicit reporting of missing voices

## Interview Logic

The protocol should be organized around three evidence layers:

1. Unprompted narrative
2. Workflow specifics
3. Data-prompted reaction to artifacts from the posting dataset

Open-ended questions must come first. Data prompts come later so participants do not simply mirror the researcher's framing.

## Core Interview Modules

### Module 1: The shift as experienced

Used with all cohorts.

Questions to cover:

- What has changed in software work or hiring in the last 2 years?
- Was there a specific moment the shift became obvious?
- Does the change feel gradual, sudden, or mostly narrative-driven?

### Module 2: Actual AI use in work

Used with seniors, juniors, and hiring-side actors with technical exposure.

Questions to cover:

- Describe a recent coding or review workflow using AI
- Which tasks did AI replace, accelerate, or create?
- Which tasks remain stubbornly human?
- What is still considered too risky to delegate?

### Module 3: The junior bar

Used with all cohorts.

Questions to cover:

- Has the bar for junior hires changed?
- If so, why?
- Is the firm actually willing to train juniors?
- Are junior candidates being evaluated on system design, deployment, ownership, or AI fluency?

### Module 4: The senior role

Used mainly with senior and hiring-side cohorts.

Questions to cover:

- Has senior work shifted away from mentoring or coordination?
- Has review, orchestration, architecture, or evaluation become more central?
- Has the team changed what it expects from a "senior engineer"?

### Module 5: JD authorship and screening

Used mainly with hiring-side actors and seniors involved in hiring.

Questions to cover:

- Who wrote the posting?
- Which bullets were copied from templates?
- Which requirements are truly screened?
- Which requirements are defensive or aspirational?
- Do AI-related requirements affect interview loops or only the written JD?

### Module 6: Meaning, anxiety, and future

Used with all cohorts.

Questions to cover:

- Does the traditional ladder still feel real?
- What does expertise mean now?
- What do people outside the field misunderstand about this moment?

## Data-Prompted Elicitation Materials

Prepare a small but sharp set of artifacts.

| Artifact | What to show | Best cohorts | Purpose |
|---|---|---|---|
| Inflated junior JD | Real entry-level posting with system design, CI/CD, AI-tool, ownership language | All cohorts | Test scope inflation |
| Paired JDs over time | Similar role from 2023-2024 vs. 2026 | Seniors, hiring-side | Make change concrete |
| Junior-share trend plot | Junior SWE share over time | Seniors, hiring-side | Compare perceived timing to data |
| Senior archetype chart | Management vs. orchestration language over time | Seniors, hiring-side | Validate senior shift |
| Posting-usage divergence chart | Posting AI mention rate vs. observed AI usage benchmark | All cohorts | Probe anticipatory restructuring |

Protocol rule:

- ask the open question first
- show the artifact second
- ask what feels real, false, missing, or overstated

## Suggested Question Bank by Cohort

### Seniors

- How has your day-to-day work changed in the last 2 years?
- What has become more important: writing code, reviewing code, system design, coordinating work, or supervising AI output?
- If you had budget for one hire, who would you hire today and why?
- Does this "entry-level" JD look real to you?
- Which listed requirements would your team actually screen for?
- Do you think juniors today are missing apprenticeship opportunities?

### Juniors

- What does the SWE job market feel like from where you sit?
- How do you use AI when coding or interviewing?
- Does this "entry-level" JD feel genuinely entry-level?
- What parts of the role feel like you are competing with AI, and what parts do not?
- Do you believe the career ladder you expected still exists?

### Hiring-side actors

- Walk me through how a JD gets written in your organization
- Which stakeholders add requirements and why?
- Have AI-related bullets been added recently? By whom?
- Are these requirements actually screened in interviews?
- Are you raising the bar because AI changed the work, or because the market got tighter, or both?
- Have you reduced willingness to hire juniors? Why?

## Analysis Plan

### Method

Reflexive thematic analysis with hybrid deductive and inductive coding.

### Deductive code families

- `junior_scope_inflation`
- `senior_archetype_shift`
- `ghost_requirement`
- `jd_authorship`
- `screened_vs_unscreened_requirement`
- `actual_ai_workflow_change`
- `anticipatory_restructuring`
- `market_tightness_non_ai`
- `career_ladder_breakdown`
- `zeitgeist_anxiety`
- `zeitgeist_opportunity`

### Cross-cohort comparison

The most publishable insights will likely come from contradictions across cohorts.

Examples:

| Theme | Senior framing | Junior framing | Hiring-side framing |
|---|---|---|---|
| Junior bar | "AI handles the basics" | "They want senior output for junior pay" | "We can be pickier in this market" |
| AI requirement | "Useful but not central" | "Feels mandatory to signal fluency" | "Sometimes added because leadership expects it" |
| Senior shift | "More review and architecture" | "Seniors feel less available to mentor" | "We need leverage, not headcount" |
| Ghost requirements | "Nice to have" | "Looks impossible" | "Some bullets are template-driven" |

### Integration with quantitative analysis

The interviews should be used to adjudicate between competing explanations for the same quantitative signal.

Example:

- Quantitative result: junior postings mention more system design
- Possible explanations:
  - real work changed
  - recruiter copied senior template
  - firms want fewer but stronger juniors
  - firms are signaling modernity by mentioning AI and ownership

The interview evidence helps separate these mechanisms.

## Practical Workflow

1. Finalize protocol after 2 pilot interviews
2. Submit IRB exempt application before formal data collection
3. Record audio with consent
4. Transcribe quickly
5. Write short analytic memos immediately after each interview
6. Code iteratively rather than waiting for all interviews to finish
7. Revise later interviews if an important mechanism emerges early
8. Conduct light member checking with a few participants

The post-interview memo step is important. It captures context and hypotheses that transcripts alone often miss.

## Timeline

| Week | Activity |
|---|---|
| Week 1 | Finalize protocol, pilot 2 interviews, build artifacts, submit IRB exempt application |
| Week 2 | Recruit aggressively, begin interviews, transcribe and memo same day |
| Week 3 | Complete most interviews, start coding, refine probes where needed |
| Week 4 | Finish coding, write cross-cohort synthesis, map mechanisms back to quantitative findings |

## How This Connects to the Revised Research Design

| Quant construct | What interviews test |
|---|---|
| Junior scope inflation | Whether inflated junior requirements are perceived as real and why they appeared |
| Senior archetype shift | Whether senior work truly shifted toward orchestration, review, and architecture |
| Posting-usage divergence | Whether AI mentions reflect actual use or anticipatory signaling |
| Ghost requirements | Whether listed requirements are screened, copied, defensive, or aspirational |
| Break analysis | Whether practitioners remember a sudden shift, a gradual shift, or no meaningful break |

## Why This Version Is Stronger

The earlier design was already good at capturing worker experience. This version is stronger because it adds the missing mechanism layer: the people who write, approve, and screen job descriptions.

That turns the qualitative component from supportive texture into direct evidence on:

- employer intent
- requirement realism
- anticipatory restructuring
- the difference between changed work and changed hiring narratives
