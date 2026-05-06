# Vocabulary List Construction — Sub-Agent Task Spec

This spec is used to construct the keyword-density vocabulary lists referenced in the
paper's Methodology > Vocabulary Lists section. One Opus 4.7 sub-agent runs the prompt
template below per topic. Outputs are merged into `vocab_lists.json` in this directory.

## Prompt template

```
You are designing a vocabulary list for keyword-density analysis in a SWE job-posting
research paper measuring how job descriptions are changing during the AI transition
(2024 → 2026, LinkedIn, US, English, ≥15 words). One analytical axis is the density of
specific conceptual markers per posting; we will count occurrences across the corpus
and track shifts over time and across role/seniority cells.

Your assigned topic: {TOPIC_NAME}
Brief as written in the paper: {TOPIC_HINT}

Produce a TWO-TIER vocabulary list:

Tier 1 — Core concepts (target 5–15):
The high-level buckets covering this topic. Each must be:
- mutually distinguishable from sibling concepts,
- collectively comprehensive of the topic's scope per the brief,
- a human-readable label, not a regex.

Tier 2 — Keyword variants grouped under each core concept:
For each core concept, exhaustively list literal strings a posting might use to express
it. A hit on any variant counts as a hit on the parent concept. Include:
- hyphen / no-hyphen / colon / space variants (1-on-1, 1:1, one-on-one, one on one)
- plurals, possessives, gerunds, verb forms (mentor, mentors, mentoring, mentorship, mentored)
- American vs British spellings (organize/organise, optimization/optimisation)
- abbreviations and expansions (PR ↔ pull request, IC ↔ individual contributor)
- common synonyms actually seen in JDs (not invented)
- compound and adjacent forms ("direct report" / "direct-report" / "direct reports")

Targets:
- ~10–40 variants per concept (more is fine if real)
- Match against case-insensitive description text with word-boundary semantics
  (assume the consumer wraps each entry in \b…\b unless you flag otherwise)

Quality bar:
- Every variant must be something a real US LinkedIn SWE JD would plausibly contain.
- Flag false-positive risks in `exclusions` (e.g., bare "lead" matches unrelated text;
  "agile" inside "fragile"; "PR" as Puerto Rico). Recommend negative-context guards.
- Be conservative on bare 2-letter abbreviations — prefer requiring context.
- If a keyword is genuinely ambiguous between two of YOUR concepts, pick one and note it.

Output STRICTLY as JSON:

{
  "topic": "<topic name>",
  "definition": "<2–3 sentence operational definition for the topic as a whole>",
  "core_concepts": [
    {
      "name": "<concept label>",
      "definition": "<one-sentence definition>",
      "keywords": ["<literal string 1>", "<literal string 2>", ...],
      "regex_notes": "<optional: word-boundary or context guards if needed>"
    }
  ],
  "exclusions": [
    {"pattern": "<string>", "reason": "<why it false-positives>", "guard": "<suggested fix>"}
  ],
  "calibration_recommendations": "<which concepts you suspect will be noisy enough to need manual spot-check on a 100-post sample before trusting trends>",
  "notes": "<edge cases, overlaps with other vocab lists in the paper, anything the analyst should know>"
}

Do not return prose outside the JSON.
```

## Topics

| Slug | TOPIC_NAME | TOPIC_HINT |
|---|---|---|
| people_management | People-management markers | direct report, team lead, supervise/oversee, 1-on-1, headcount, performance review |
| orchestration | Orchestration | specs, decomposition, architecture, workflow design, context engineering, agent harnesses, repository instructions, memory, multi-agent coordination |
| verification | Verification | tests, validation, evals, proof, CI, reproducibility, screenshots/logs, manual verification, quality gates, code review focus, static analysis, compliance, post-deployment observability |
| mentorship | Mentorship markers | mentorship, code reading, debugging, paired work, onboarding, architecture review, explicit learning |
| performance | Performance optimization & deep technical understanding | demand for low-level perf work, profiling, latency/throughput, systems-level depth, "deep understanding of X" framing |
| process_scaffolding | Process-scaffolding markers | agile, scrum, requirements, V&V, specification, coordinate, schedule |
| legacy_stack | Legacy-stack markers | .NET, COBOL, mainframe, VMware, Active Directory |
| context_infrastructure | Context infrastructure | documentation quality, telemetry, data integration, system understanding, product/business awareness, cross-functional language |

## Run mechanics

- One Agent call per topic (`subagent_type: general-purpose`, `model: opus`), all in a single parent message → 8 parallel sub-agents.
- Each sub-agent writes its JSON to `/tmp/vocab_<slug>.json` and returns a one-line confirmation.
- The parent reads all 8 files, validates JSON, merges to `vocab_lists.json`, and deletes `/tmp/vocab_*.json`.

## Downstream consumption

`vocab_lists.json` schema:

```json
{
  "topics": {
    "<slug>": { /* per-topic JSON, exactly as the sub-agent emitted */ }
  },
  "generated_at": "<ISO-8601>",
  "spec_version": "<spec hash or version>"
}
```

A keyword hit on any variant under a concept counts as one hit on the parent concept; a
hit on any concept under a topic counts as one hit on the parent topic. Cross-list
overlaps (e.g., "code review" in both Verification and Mentorship) are reconciled
manually after merge by inspecting the `notes` field of each topic.
