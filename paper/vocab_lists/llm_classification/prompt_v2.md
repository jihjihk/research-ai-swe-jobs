# Prompt v2 (frozen)

Frozen artifact. Do not edit. To revise, copy to `prompt_v3.md` and bump callers.

## Changes from v1

Compressed per-label definitions to **trigger phrases + NOT clauses** (verb-form, observable signals). Closing reminder repeats the explicit-only rule. `legacy_stack` intentionally broadened to cover any-version Java / .NET / etc. as old-paradigm enterprise frameworks. See `PILOT_REPORT.md` for the diagnostic data that motivated each change.

## System message

```
You tag SWE job postings with which of 8 themes are explicitly named as responsibilities, requirements, or skills. Boilerplate, passing mentions, and company descriptions are not evidence.

Themes:
- people_management: direct reports, performance reviews, hiring/firing authority, headcount, 1:1 cadence. NOT bare "lead a team" / "tech lead" / "lead engineer".
- orchestration: authoring specs, ADRs, RFCs, or design docs; decomposing work for engineers or AI agents; multi-agent or context-engineering systems. NOT design patterns as required knowledge (MVC, MVVM); NOT "architecture experience" without authoring.
- verification: CI/CD pipelines, named test frameworks, code-review processes, evals, observability for regressions, compliance/audit, static analysis. NOT generic "writes tests".
- mentorship: mentoring, teaching, growing, or onboarding other engineers; pair programming; knowledge transfer. NOT managing a team (that's people_management).
- performance: profiling, latency, throughput, kernel/network internals, low-level optimization, "deep understanding of [a technical area]". NOT "high-performing team"; NOT "expert in [tool]" as recruiter fluff.
- process_scaffolding: agile, scrum, sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack: required experience with old-paradigm enterprise frameworks regardless of version — Java / Java EE, .NET / .NET Framework, ASP.NET, COBOL, mainframe, VMware / vSphere, Active Directory, BizTalk, ColdFusion, and similar.
- context_infrastructure: authoring runbooks, ADRs, RFCs, dashboards / SLOs; telemetry hygiene; schema documentation; technical writing. NOT generic "cross-functional collaboration" or "communication skills".

Reminder: tag only what the posting explicitly names. Multi-label allowed. Empty array means none. Output JSON array of slugs, no commentary.
```

## User message template

Unchanged from v1.

```
Title: {title}
Description: {description_core_llm}
```

## Structured output schema

Unchanged from v1 — same 8-enum constraint:

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

## Rationale (per-label diffs from v1)

| Label | v1 → v2 change | Why |
|---|---|---|
| `people_management` | Added `NOT bare "lead a team" / "tech lead" / "lead engineer"` | Smaller models conflated IC technical-leadership with formal management authority. |
| `orchestration` | Replaced abstract "system architecture" with verb-form "authoring specs/ADRs/RFCs/design docs"; added NOT clauses for design-pattern knowledge and bare "architecture experience" | 24% inter-model disagreement on v1 — the noun "architecture" was triggering on required-knowledge mentions (MVC, MVVM) and on senior-IC postings without authoring activity. |
| `verification` | Added `NOT generic "writes tests"` | Bare "writes tests" wasn't a discriminating signal — every SWE writes tests. v2 requires named processes/frameworks. |
| `mentorship` | Added `NOT managing a team (that's people_management)` | Cross-label disambiguation; helps the LLM with the IC-vs-manager line. |
| `performance` | Replaced `OR explicit demands for deep technical depth` with concrete `"deep understanding of [a technical area]"`; added NOT clauses for "high-performing team" and "expert in [tool]" recruiter fluff | The OR-construction in v1 conflated low-level perf work with generic depth-claim language; the regex calibration showed the same failure mode. |
| `process_scaffolding` | Unchanged | Already concrete; 12% disagreement was driven by boundary cases with verification, not by the definition itself. |
| `legacy_stack` | Replaced examples-list with broader "old-paradigm enterprise frameworks regardless of version" | Per-author scoping decision: any-version Java / .NET / etc. count. v1's implicit modern/legacy line was inconsistent across model tiers (16% disagreement, nano κ = 0.24 vs mini). |
| `context_infrastructure` | Replaced abstract noun list with verb-form authoring activities; added NOT clauses for "cross-functional collaboration" and "communication skills" fluff | Same drift as orchestration — abstract concepts like "documentation" and "telemetry" attracted boilerplate matches. |

## Closing reminder

The "Reminder:" line at the bottom is new. v1 had the explicit-only rule once at the top of the system message; per-label decisions weren't always re-applying it. Repeating in compact form near the model's reasoning point reinforces it without adding much length.

## Token budget

System prompt: ~340 tokens (was ~280 in v1). User message and output schema unchanged. Per-call paid input rises ~60 tokens; for a 60k-posting × 3-rep full-corpus run on `gpt-5.4-mini`, that's ~$2-5 of additional cost — negligible.

## What was rejected

- Few-shot examples (would help nano substantially but inflate tokens 3-4×; mini already passes the bar without them).
- Splitting `orchestration` into two labels (would break comparability with the 8-topic regex layer).
- Per-label confidence scoring (out of scope for binary multi-label).

## Calibration not yet performed

`prompt_v2.md` has not been re-piloted yet. The plan: run the same 25-posting × `gpt-5.4-mini` × 3-rep design under v2, compare per-label inter-model agreement and disagreement-posting count to v1's pilot. Promote v2 if `orchestration` and `legacy_stack` disagreement drop and no new label regresses.
