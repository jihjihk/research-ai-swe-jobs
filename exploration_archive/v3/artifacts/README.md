# T25: Interview Elicitation Artifacts

**Date:** 2026-04-05
**Purpose:** Visual and textual artifacts for RQ4 semi-structured interviews with SWE hiring managers and practitioners. All findings use CORRECTED numbers from the full exploration (T01-T23).

---

## Artifact Index

### 1. Entry-Level JD Examples with Scope Indicators (Tables)

**Files:**
- `exploration/tables/T25/entry_2026_scope_jds.csv` -- 20 entry-level 2026 postings showing validated scope expansion (mentor, coach, project management, stakeholder terms; YOE <= 3; non-aggregator)
- `exploration/tables/T25/entry_2024_contrast_jds.csv` -- 20 entry-level 2024 postings (arshkon) for temporal contrast

**Interview use:** Show 2024 vs 2026 entry-level JDs side by side. Ask: "This 2024 posting asked for 2 YOE and Python/SQL. This 2026 posting asks for 2 YOE plus mentoring, stakeholder engagement, and AI/ML familiarity. Does this match what you see?" Note: the management indicator increase is moderate (+4-10pp using validated patterns), not the +31pp initially reported.

### 2. Paired JDs Over Time (Company Panel)

**File:** `exploration/tables/T25/company_ai_entry_orthogonality.csv` -- 463 companies with postings in both periods, showing AI adoption change and entry-share change.

**Interview use:** Select company pairs illustrating different strategies:
- **AI-forward:** Oracle (0% -> 100% AI), Deloitte (0% -> 94% AI), KPMG (0% -> 97% AI). These companies massively adopted AI requirements.
- **Traditional hold:** Google (96% -> 96% AI, already high), Boeing (13% -> 11% AI, stable low), Northrop Grumman (1% -> 2% AI, defense/traditional).
- **Scope inflators:** Companies with large description length growth but modest AI change (see T16 cluster C0: AMD, Adobe, Accenture).

Ask: "Your company went from X to Y on AI requirements. What drove that? Did you change actual job content, or posting templates?"

### 3. Junior-Share Trend Visualization

**File:** `exploration/figures/T25/junior_share_trend.png`

**Description:** Entry-level SWE posting share plotted from April 2024 to March 2026, annotated with major AI model release dates (GPT-4o, Claude 3.5 Sonnet, o1, DeepSeek V3, Claude 4 Opus, Claude Opus 4.5). Uses arshkon-only as 2024 baseline with seniority_native. Includes prominent caveat that direction depends on seniority operationalization.

**Interview use:** Show the decline trend with AI model releases. Ask: "Entry-level SWE postings declined from ~22% to ~14% of known-seniority postings. Eight major AI models were released in between. Do you think these are connected? Our data shows no correlation at the firm level -- companies that adopted AI didn't cut junior roles more than others."

### 4. Senior Archetype Chart

**File:** `exploration/figures/T25/senior_archetype_chart.png`

**Description:** Two-panel figure. Left: management language density by seniority level (2024 vs 2026), showing universal expansion and -23% at director level. Right: senior sub-archetype shares (Low-touch Generalist, People Manager, Technical Orchestrator, Strategic Leader), showing orchestrator growth +29% relative and strategic leader decline -20% relative.

**Interview use:** Ask: "We see director-level postings using less people-management language and more technical orchestration language. Meanwhile a 'technical orchestrator' archetype is growing -- senior ICs focused on system design, AI workflows, and code review rather than team management. Does this match your organizational reality?"

### 5. AI Divergence Chart

**File:** `exploration/figures/T25/ai_divergence_chart.png`

**Description:** Bar chart comparing posting AI requirement rates vs estimated developer usage rates, by category (AI-any, AI-tool, AI-domain). Key finding: requirements lag usage (~41% posting vs ~75% usage in 2026), but the gap narrowed from -45pp to -34pp. AI-as-domain may slightly overshoot specialist rates.

**Interview use:** Ask: "About 41% of SWE postings now mention AI, but an estimated 75% of developers actually use AI tools daily. Why do you think posting requirements lag actual usage? Are AI skills assumed rather than listed? Or do some roles genuinely not need AI? Also -- do you think the 23% of postings mentioning specific tools like Copilot or Claude reflect real skill requirements, or just keyword optimization?"

### 6. AI-Entry Orthogonality Scatter

**File:** `exploration/figures/T25/ai_entry_orthogonality.png`

**Description:** Company-level scatter plot of AI prevalence change vs entry-level share change (2024 to 2026). Shows null correlation (r = -0.019, p = 0.683, n = 463 companies). Includes explanation box noting the metro-level null as well (r = -0.04).

**Interview use:** This is the most striking finding for interviewees. Present it as: "At the market level, AI adoption surged and junior hiring declined simultaneously. But when we look within individual companies, these trends are completely unrelated. Companies that adopted AI the most aggressively did NOT cut junior roles. What do you make of this? If AI isn't directly causing junior hiring reduction within firms, what else is going on?"

---

## Key Interview Themes

Based on the exploration findings, the three highest-priority interview themes are:

### Theme A: The Orthogonality Puzzle
Why do AI adoption and junior hiring changes co-occur at the market level but not within firms? Is the connection indirect (mediated through company composition, investor expectations, or market sentiment)? Or is the relationship delayed -- will firms that adopted AI eventually restructure their junior hiring?

### Theme B: Are AI Requirements Real Hiring Bars or Signaling?
Our data shows AI requirements are less hedged than traditional requirements (20% vs 30% aspirational language). But does listing "LLM experience" on a junior posting actually filter candidates? Or is it aspirational? Do interview processes test for AI skills?

### Theme C: The Domain Recomposition Experience
ML/AI engineering grew from 4% to 27% of SWE postings. Frontend/Web contracted from 41% to 24%. How are practitioners experiencing this shift? Are frontend engineers retraining for ML? Are new hires coming in with ML skills by default? Is the domain shift real (new work) or cosmetic (rebranding existing work)?

---

## Usage Notes

1. All figures are 150 DPI PNGs suitable for screen display during interviews.
2. The company panel data (CSV) allows the interviewer to select company-specific examples relevant to each interviewee's employer.
3. The JD tables include UIDs for full-text retrieval from `data/unified.parquet` if deeper examples are needed.
4. All statistics use CORRECTED values. The management indicator is +4-10pp (not +31pp from T11's uncorrected broad patterns).
