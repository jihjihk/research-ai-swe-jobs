# Prompt: combined v3 (frozen)

Frozen artefact. Two-axis multi-label classification combining the v2 skill themes and the v1 role families into a single prompt with two output arrays. Used as the **combined variant** in the v3 pilot, alongside `prompt_skill_v2.md` and `prompt_role_family_v1.md`.

The v3 pilot tests whether the combined variant matches the standalone variants per axis. If yes → ship combined (one call per posting, half the cost). If no → ship two separate prompts.

## System message

```
You tag SWE job postings on TWO axes: skill themes (8) and role families (17). Both are multi-label. Tag only what the posting explicitly names as a responsibility, requirement, or skill — not boilerplate, passing mentions, or company descriptions.

SKILL THEMES — what work the role does:
- people_management: direct reports, performance reviews, hiring/firing authority, headcount, 1:1 cadence. NOT bare "lead a team" / "tech lead" / "lead engineer".
- orchestration: authoring specs, ADRs, RFCs, or design docs; decomposing work for engineers or AI agents; multi-agent or context-engineering systems. NOT design patterns as required knowledge (MVC, MVVM); NOT "architecture experience" without authoring.
- verification: CI/CD pipelines, named test frameworks, code-review processes, evals, observability for regressions, compliance/audit, static analysis. NOT generic "writes tests".
- mentorship: mentoring, teaching, growing, or onboarding other engineers; pair programming; knowledge transfer. NOT managing a team (that's people_management).
- performance: profiling, latency, throughput, kernel/network internals, low-level optimization, "deep understanding of [a technical area]". NOT "high-performing team"; NOT "expert in [tool]" as recruiter fluff.
- process_scaffolding: agile, scrum, sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack: required experience with old-paradigm enterprise frameworks regardless of version — Java / Java EE, .NET / .NET Framework, ASP.NET, COBOL, mainframe, VMware / vSphere, Active Directory, BizTalk, ColdFusion, and similar.
- context_infrastructure: authoring runbooks, ADRs, RFCs, dashboards / SLOs; telemetry hygiene; schema documentation; technical writing. NOT generic "cross-functional collaboration" or "communication skills".

ROLE FAMILIES — the engineering archetype:
- software_engineer_general: residual fallback. NEVER use alongside another family — only when no other fits.
- frontend_web: user-facing web/mobile UI/UX work. Full-stack roles default here.
- backend_api: server-side logic, APIs, databases, services.
- mobile: native or cross-platform mobile apps (iOS, Android, React Native, Flutter).
- embedded: firmware, IoT, hardware-near software, RTOS, FPGA.
- data_engineer: data pipelines, warehouses, ingestion infrastructure.
- ml_engineer: pre-LLM ML pipelines, model training, productionising ML systems, MLOps.
- ai_llm_engineer: LLM-based or agent systems — prompt engineering, RAG, foundation-model integration, multi-agent.
- devops_sre_platform: CI/CD, infrastructure-as-code, reliability, platform engineering, SRE.
- security: application or infrastructure security.
- qa_test: software quality, testing, validation as the role's primary focus (QA Engineer, SDET, Test Automation).
- solutions_field: deploys / integrates with customers (pre- and post-sales) — Solutions Engineer, Field Applications, Forward-Deployed.
- legacy_specialist: role's primary purpose is maintaining a legacy stack (Mainframe, COBOL, classic .NET, Salesforce, ServiceNow, SAP/ABAP, Oracle Forms). Not just one of many required skills.
- data_analytics: analytical, operational, or decision-support outputs (Data Scientist, Data Analyst, Analytics Engineer). Not production ML pipelines.
- research: applied or basic research bridging academia and product (Research Scientist, Applied Scientist, Research Engineer).
- infra_ops_admin: operates/administrates databases, servers, networks rather than authoring software (DBA, sysadmin, network engineer, cloud admin).
- people_manager: engineering management focused on people, hiring, team operations (Engineering Manager). Excludes director/VP/C-level.

Reminder: tag only what the posting explicitly names. Multi-label allowed on both axes. Output JSON with two arrays — `skill_themes` and `role_families`. Empty arrays mean none. No commentary.
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
    },
    "role_families": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "software_engineer_general",
          "frontend_web",
          "backend_api",
          "mobile",
          "embedded",
          "data_engineer",
          "ml_engineer",
          "ai_llm_engineer",
          "devops_sre_platform",
          "security",
          "qa_test",
          "solutions_field",
          "legacy_specialist",
          "data_analytics",
          "research",
          "infra_ops_admin",
          "people_manager"
        ]
      }
    }
  },
  "required": ["skill_themes", "role_families"],
  "additionalProperties": false
}
```

## Token budget

System prompt ≈ 570 tokens (vs 340 for skill-only v2 and ~340 for role-family-only v1). Per-call paid input rises ~230 tokens compared to either standalone variant. For a 60k × 3 reps full-corpus run on `gpt-5.4-mini`: ~$15-30 extra vs running just the combined variant. If we instead had to run BOTH standalone variants per posting, the cost would be roughly 2× the combined variant's cost.

## Decision criterion

If, on the v3 pilot, **per-axis Jaccard between standalone-variant majority and combined-variant majority is ≥ 0.95**, the combined variant is safe and we ship it. If 0.85-0.95, document the tradeoff. If <0.85, ship two separate prompts.
