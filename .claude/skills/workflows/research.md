# Workflow: Technical Research

## Purpose

Targeted methodological research for the SWE labor study. Every research task connects back to a specific RQ, hypothesis, or methodological decision in the project.

Use this for questions like:
- How do other papers normalize job posting time series with varying scrape volumes?
- What statistical tests detect structural breaks with limited post-break data?
- How do Lightcast/Burning Glass studies handle seniority classification?
- What validation approaches do scraping-based labor studies use?

This is NOT for general literature review (that lives in `docs/sources.txt`) or for exploratory "learn about X" tasks. Every research task should produce actionable methodological guidance.

---

## Project Context

Before any research, read these files to understand the study's framework:
- `docs/research-design-h1-h3.md` — research questions and hypotheses
- `docs/validation-plan.md` — planned analytical approaches
- `docs/sources.txt` — known sources (do not duplicate)
- `docs/research-*.md` — prior research syntheses (avoid re-covering the same ground)

---

## Folder Structure

```
research/                          # Scratch workspace (gitignored)
├── {topic-slug}/
│   ├── notes.md                   # Working notes, raw search results, quotes
│   └── sources/                   # Saved PDFs, article summaries
│
docs/                              # Final outputs (git tracked)
├── research-{topic-slug}.md       # Clean synthesis document
```

- `research/` is intermediate work — raw notes, search dumps, saved sources. Gitignored.
- `docs/research-*.md` is the deliverable — a clean synthesis. Committed to git.

---

## Process

### Phase 1: Scope

1. Read the project context files listed above.
2. Parse the user's topic into a specific question.
3. Identify which RQ(s) or methodological decision this supports.
4. Decompose into 3-5 sub-questions, each independently searchable.
5. Present the sub-questions to the user for approval before searching.

Do not proceed to Phase 2 without user confirmation on scope.

### Phase 2: Parallel Search

Spawn **3 subagents** in parallel, organized by source domain. Each subagent gets:
- The full research question and all sub-questions
- Its specific source domain and search strategies
- Explicit instructions on what to extract (methods, not findings)

**Important**: Give each subagent ALL sub-questions, not one per agent. Organizing by source domain (academic vs. practitioner vs. critical) produces better results than organizing by sub-question, because each domain has different search patterns and quality signals.

#### Agent 1: Academic sources

Focus on peer-reviewed papers, NBER/IZA working papers, and institutional reports (OECD, ILO, BLS).

Search strategies:
```
site:scholar.google.com "{topic}"
site:arxiv.org "{topic}"
site:ssrn.com "{topic}"
site:nber.org "{topic}"
"{topic}" filetype:pdf
"{topic}" "working paper" OR "preprint"
"{topic}" methodology OR "data quality"
```

Tell this agent to focus on **methods sections** — how researchers actually handled the data, not their findings. Extract specific techniques (e.g., "60-day dedup window" not "they deduplicated").

#### Agent 2: Practitioner/implementation sources

Focus on code, tools, tutorials with implementation details, and blog posts by people who actually did the work.

Search strategies:
```
site:github.com "{topic}" implementation OR notebook OR pipeline
site:towardsdatascience.com "{topic}"
site:stackoverflow.com "{topic}"
"{topic}" "in practice" OR tutorial OR implementation
"{topic}" python OR library OR package
```

Tell this agent to find **concrete tools, thresholds, and code** — not conceptual descriptions. A GitHub repo with a working pipeline is worth more than 10 blog posts.

#### Agent 3: Validity threats and critical sources

Focus on what goes wrong, what reviewers reject, and known pitfalls.

Search strategies:
```
"{topic}" limitation OR caveat OR pitfall OR bias
"{topic}" "doesn't work" OR "failed" OR alternative
"{topic}" validity OR representativeness OR "selection bias"
"{topic}" "reviewer" OR "referee report" OR "limitations section"
```

Tell this agent to focus on **specific failure modes and their mitigations** — not vague warnings. "Ghost jobs are 18-27% of postings (CRS 2025)" is useful. "There may be data quality issues" is not.

#### Subagent prompting guidelines

Each subagent prompt must include:
1. The full research question and context (which RQs it supports)
2. All sub-questions (not just its assigned domain)
3. 6-10 specific search queries to run
4. Explicit instructions on output format: for each source, return title/authors/year, URL, specific methodological claims, and relevance to our study
5. Instruction to focus on methods/techniques, not findings

Do NOT:
- Split sub-questions across agents (this fragments coverage)
- Give vague prompts like "search for academic sources on X" (specify exact queries)
- Forget to tell agents about known sources to avoid duplicating

### Phase 3: Synthesize

Skip the "sequential deepening" phase from the original design — in practice, the parallel search with well-prompted agents produces sufficient coverage. Only do follow-up searches if a specific promising thread emerges that none of the agents covered.

Combine findings into `docs/research-{topic-slug}.md` using this format:

```markdown
# Research: {Topic Title}

Date: {YYYY-MM-DD}
Context: {Which RQ/hypothesis/methodological decision this supports}

## Question

{The specific question and why it matters for our study}

## Sub-questions explored

1. {sub-question}
2. {sub-question}
...

## Findings

### {Sub-question 1}

{Synthesized answer drawing on multiple sources. Include specific methods,
formulas, parameter choices, and implementation details where relevant.}

### {Sub-question 2}

{...}

## Recommendations for our study

Split into priority tiers:

### Immediate actions
{Things to add to the notebook now — specific, implementable steps}

### Methodology section additions
{What to write in the paper's methods/limitations section}

### Nice-to-have
{More rigorous approaches if time permits}

## Key sources

{Annotated list — for each source:}
- **{Title}** — {Author(s)}, {Year}
  - Relevance: {one line on why this matters for us}
  - URL: {link}

## Open questions

{What remains unresolved or needs further investigation}
```

### Phase 4: Present

Show the user the synthesis and ask:
1. Does this answer the question?
2. Are the recommendations actionable?
3. Anything to dig deeper on?

Do NOT clean up `research/{topic-slug}/` automatically — the user may want to reference intermediate notes later. Only clean up if explicitly asked.

---

## Quality Checklist

Before presenting the synthesis:
- [ ] Connected to a specific RQ or methodological decision
- [ ] Sub-questions were scoped and approved by user
- [ ] Academic sources consulted (not just blog posts)
- [ ] Practitioner/implementation sources found where relevant
- [ ] Limitations and pitfalls of recommended approach documented
- [ ] Alternative approaches mentioned (not just the first one found)
- [ ] Recommendations are concrete and tiered (immediate / methods section / nice-to-have)
- [ ] Known project sources not duplicated
- [ ] Final synthesis is in `docs/research-{topic-slug}.md`

---

## Lessons Learned

These are observations from running this workflow. Update this section as more research tasks are completed.

- **Organize agents by source domain, not by sub-question.** Splitting sub-questions across agents fragments coverage. Each agent should see all sub-questions but search its own source domain.
- **Be very specific in agent prompts.** Include 6-10 actual search queries, not just "search for X." Tell agents exactly what to extract (methods, thresholds, code) vs. what to skip (findings, abstractions).
- **Skip the sequential deepening phase.** Well-prompted parallel agents with 20+ search queries each produce sufficient coverage. Follow-up is rarely needed.
- **The skill is not auto-invocable.** It must be run manually by following the process. This is fine — the value is in the structured process and output format, not in automation.
- **Tiered recommendations work better than a flat list.** Splitting into "immediate actions / methodology section / nice-to-have" makes the output directly actionable.
- **Don't auto-delete intermediate notes.** The user may want to revisit raw search results later.
- **Prior research outputs should be checked.** Add `docs/research-*.md` to the context files list to avoid re-covering ground from previous research tasks.
