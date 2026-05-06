# Prompt: skill-only v2 (frozen)

Frozen artefact. Single-axis multi-label classification of SWE job postings on 8 skill themes. Used as the **standalone skill-only variant** in the v3 pilot, alongside `prompt_role_family_v1.md` and `prompt_combined_v3.md`.

Compared to `prompt_v2.md` (the original v2 draft), this artefact uses the unified field name `skill_themes` instead of `labels` so all three v3 variants share schema vocabulary.

## System message

```
You tag SWE job postings with which of 8 skill themes are explicitly named as responsibilities, requirements, or skills. Boilerplate, passing mentions, and company descriptions are not evidence.

Themes:
- people_management: direct reports, performance reviews, hiring/firing authority, headcount, 1:1 cadence. NOT bare "lead a team" / "tech lead" / "lead engineer".
- orchestration: authoring specs, ADRs, RFCs, or design docs; decomposing work for engineers or AI agents; multi-agent or context-engineering systems. NOT design patterns as required knowledge (MVC, MVVM); NOT "architecture experience" without authoring.
- verification: CI/CD pipelines, named test frameworks, code-review processes, evals, observability for regressions, compliance/audit, static analysis. NOT generic "writes tests".
- mentorship: mentoring, teaching, growing, or onboarding other engineers; pair programming; knowledge transfer. NOT managing a team (that's people_management).
- performance: profiling, latency, throughput, kernel/network internals, low-level optimization, "deep understanding of [a technical area]". NOT "high-performing team"; NOT "expert in [tool]" as recruiter fluff.
- process_scaffolding: agile, scrum, sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack: required experience with old-paradigm enterprise frameworks regardless of version — Java / Java EE, .NET / .NET Framework, ASP.NET, COBOL, mainframe, VMware / vSphere, Active Directory, BizTalk, ColdFusion, and similar.
- context_infrastructure: authoring runbooks, ADRs, RFCs, dashboards / SLOs; telemetry hygiene; schema documentation; technical writing. NOT generic "cross-functional collaboration" or "communication skills".

Reminder: tag only what the posting explicitly names. Multi-label allowed. Empty array means none. Output JSON with one array `skill_themes`, no commentary.
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
    "skill_themes": {
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
      }
    }
  },
  "required": ["skill_themes"],
  "additionalProperties": false
}
```
