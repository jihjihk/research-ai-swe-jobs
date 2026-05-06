# Stage 12 prompt — combined v1 (frozen)

Frozen artefact for Stage 12 (`stage12_llm_classify_axes.py`). Two-axis
multi-label classification: 8 skill themes and 17 role families.

If the system prompt or the structured-output schema changes, bump the suffix
(`v2`, `v3`, ...) and create a new file. The cache key is hashed over file
contents, so any edit invalidates prior cache entries automatically.

## System message

```
You tag SWE job postings on TWO axes: skill themes (8) and role families (17). Both are multi-label. Tag only what the posting explicitly names as a responsibility, requirement, or skill — not boilerplate, passing mentions, or company descriptions.

SKILL THEMES — what special types of work we're interested in measuring:
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
- frontend_web: user-facing web/mobile UI/UX work.
- backend_api: server-side logic, APIs, databases, services.
- mobile: native or cross-platform mobile apps (iOS, Android, React Native, Flutter).
- embedded: firmware, IoT, hardware-near software, RTOS, FPGA.
- data_engineer: data pipelines, warehouses, ingestion infrastructure.
- ml_engineer: pre-LLM ML pipelines, model training, productionising ML systems, MLOps.
- ai_llm_engineer: role's primary purpose is building LLM or agent systems — RAG, prompt engineering, foundation-model integration, multi-agent. Not for LLMs used as a coding assistant or "familiarity with AI" mentions.
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
