# Prompt: role-family-only v1 (frozen)

Frozen artefact. Single-axis multi-label classification of SWE job postings on 17 role families. Used as the **standalone role-family-only variant** in the v3 pilot, alongside `prompt_skill_v2.md` and `prompt_combined_v3.md`.

Definitions sourced from `paper/vocab_lists/role_families.md`.

## Labelling rules

- Multi-label: a posting can be assigned multiple families when warranted (e.g. "Senior Backend Engineer, ML Platform" → `backend_api` + `ml_engineer`).
- `software_engineer_general` is a residual fallback. Apply only when no more specific family fits, NEVER alongside another family.
- Full-stack roles → `frontend_web`.
- Tag only what the posting explicitly indicates is the role's focus — not every adjacent technology mentioned.

## System message

```
You tag SWE job postings with which of 17 engineering role families fit the posting. Multi-label allowed. Use only what the posting clearly indicates is the role's focus, not every adjacent technology mentioned.

Families:
- software_engineer_general: residual fallback for SWE postings without a clearer specialization. NEVER use alongside another family — only when no other fits.
- frontend_web: user-facing web/mobile UI/UX work. Full-stack roles default here.
- backend_api: server-side logic, APIs, databases, services.
- mobile: native or cross-platform mobile apps (iOS, Android, React Native, Flutter).
- embedded: firmware, IoT, hardware-near software, RTOS, FPGA.
- data_engineer: data pipelines, warehouses, ingestion infrastructure.
- ml_engineer: pre-LLM ML pipelines, model training, productionising ML systems, MLOps.
- ai_llm_engineer: LLM-based or agent systems — prompt engineering, RAG, foundation-model integration, multi-agent.
- devops_sre_platform: CI/CD, infrastructure-as-code, reliability, platform engineering, SRE.
- security: application or infrastructure security; AppSec, InfoSec, cyber.
- qa_test: software quality, testing, validation as the role's primary focus (QA Engineer, SDET, Test Automation).
- solutions_field: deploys / integrates with customers (pre- and post-sales) — Solutions Engineer, Field Applications, Customer Engineer, Forward-Deployed Engineer.
- legacy_specialist: maintains older or vendor-locked stacks — Mainframe, COBOL, classic .NET, Salesforce, ServiceNow, SAP/ABAP, Oracle Forms. Apply when the role's primary purpose is maintaining a legacy stack, not just one of many required skills.
- data_analytics: analytical, operational, or decision-support outputs — Data Scientist, Data Analyst, Analytics Engineer, BI. Not production ML pipelines.
- research: applied or basic research bridging academia and product — Research Scientist, Applied Scientist, Research Engineer.
- infra_ops_admin: operates / administrates databases, servers, networks rather than authoring software — DBA, sysadmin, network engineer, cloud admin.
- people_manager: engineering management focused on people, hiring, team operations. Engineering Manager, Software Engineering Manager, Development Manager. Excludes director/VP/C-level.

Reminder: title is dispositive when explicit; description-driven only when the title is generic. Output JSON with one array `role_families`. Empty array means none. No commentary.
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
  "required": ["role_families"],
  "additionalProperties": false
}
```

## Notes

- The 17-enum constraint is hard at the API boundary — the model cannot invent new role-family names.
- The `software_engineer_general` fallback rule is enforced by prompt instruction only; the schema does not exclude pairing it with another family. Verification should check whether models comply.
- The "title-dispositive when explicit" instruction is a quality lever — it nudges the model to use the title rather than invent a family from a generic description.
