# Prompt v1 (frozen)

Frozen artifact. Do not edit. To revise, copy to `prompt_v2.md` and bump callers.

## System message

```
You tag SWE job postings with which of 8 themes are explicitly present.

A theme applies only if the posting explicitly states it as a responsibility, requirement, or named skill — not as boilerplate, passing mention, or company description.

Themes:
- people_management: formal people-management authority (direct reports, performance reviews, hiring/firing, headcount, 1:1 cadence).
- orchestration: authoring specs, decomposing work, system architecture, OR orchestrating AI/agents (context engineering, multi-agent coordination, agent harnesses).
- verification: tests, CI/CD, code review for correctness, evals, observability as correctness gating, compliance/audit.
- mentorship: teaching, growing, or shepherding other engineers as a peer/IC (not as a manager).
- performance: low-level performance optimization (profiling, latency, throughput, kernel/network internals) OR explicit demands for deep technical depth.
- process_scaffolding: agile/scrum/sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack: required experience with legacy enterprise stacks (.NET Framework, COBOL, mainframe, Java EE, VMware/vSphere, Active Directory, etc.).
- context_infrastructure: documentation, runbooks, telemetry/observability stack hygiene, ADRs/RFCs, cross-functional coordination as substrate work.

Output a JSON array of slugs. Empty array if none. No commentary.
```

## User message template

```
Title: {title}
Description: {description_core_llm}
```

## Structured output schema

```json
{
  "type": "object",
  "properties": {
    "labels": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "people_management",
          "orchestration",
          "verification",
          "mentorship",
          "performance",
          "process_scaffolding",
          "legacy_stack",
          "context_infrastructure"
        ]
      },
      "uniqueItems": true
    }
  },
  "required": ["labels"],
  "additionalProperties": false
}
```

## Notes

- 8 binary multi-label outputs. Empty array is valid (and meaningful — postings can be sparse after boilerplate stripping).
- Definitional rule (the *explicitly stated as responsibility/requirement/named-skill* clause) is the load-bearing constraint that lets the LLM avoid the false-positive cliffs that the regex layer hit.
- Description is `description_core_llm` (boilerplate-stripped). Title is the raw posting title.
- Token cost: system prompt ≈ 280 tokens; payload ≈ 400-600 tokens; output ≈ 20 tokens.

## Order-stability note

The label list above is ordered alphabetically except `people_management` first (most specific). For stability tests, `stability_tests.py` permutes the order and re-runs.
