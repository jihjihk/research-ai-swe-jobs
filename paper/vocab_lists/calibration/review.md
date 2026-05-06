# Vocab Lists — Calibration Review

_Generated 2026-05-06T00:50:12+00:00._


Calibration sample: **19,433 SWE postings** stratified across `kaggle_arshkon/2024-04`, `kaggle_asaniczka/2024-01`, `scraped/2026-04`, matched against `description_core_llm`.


## Executive summary

- Input vocabulary: **3,303 keywords** across **88 core concepts** in **8 topics**.
- Recommended **drops:** 893 keywords (~27% of input — mostly zero-hit entries).
- Recommended **guards:** 268 keywords with high false-positive risk visible in example matches.
- Recommended **additions:** 86 new keywords surfaced from corpus inspection.
- **Cross-list collisions:** 217 keywords appear in 2+ topics; 6 reconciliation rules proposed below.

### Per-topic state

| Topic | Concepts | Keywords | Zero-hit | Drop | Guard | Add | Concept hit-rate range |
|---|---:|---:|---:|---:|---:|---:|---|
| **people_management** | 10 | 247 | 99 | 111 | 18 | 12 | 0.0% — 23.9% |
| **orchestration** | 10 | 312 | 93 | 108 | 9 | 14 | 0.0% — 54.9% |
| **verification** | 9 | 432 | 75 | 84 | 22 | 14 | 9.7% — 40.8% |
| **mentorship** | 11 | 474 | 141 | 203 | 90 | 12 | 1.2% — 29.1% |
| **performance** | 14 | 582 | 191 | 80 | 28 | 14 | 0.7% — 38.9% |
| **process_scaffolding** | 11 | 480 | 62 | 127 | 40 | 4 | 9.3% — 53.1% |
| **legacy_stack** | 12 | 300 | 73 | 79 | 29 | 8 | 0.6% — 12.6% |
| **context_infrastructure** | 11 | 476 | 98 | 101 | 32 | 8 | 2.4% — 34.2% |

## Per-topic findings

### People-management markers (`people_management`)

The vocabulary correctly identifies a small but real people-management signal in SWE postings (~24% of postings hit at least the Coaching/mentoring concept), but two concepts are deeply contaminated and must not be used as standalone people-management proxies. 'Supervisory verbs' is dominated by false positives: 'supervision' overwhelmingly matches 'minimal/limited supervision' (autonomy boilerplate), 'oversee/oversight' matches technical scope ('oversee implementations / cloud systems / change control'), and 'supervised/supervising' matches ML jargon ('supervised learning'). 'Team-size signals' is also broken: bare 'team of' fires 1,527 times but mostly in IC contexts ('part of a team of engineers'), and 'team leadership' bundles formal-mgmt phrases with senior-IC titles like 'tech lead' / 'lead software engineer' that frequently denote IC influence. Roughly 99 of 247 keywords have zero hits (40%), concentrated in the rarely-stated concepts (Termination authority, Direct reports plurals, 1:1 cadence variants, structured PIP/360 vocabulary). The high-precision concepts the paper can rely on are 'People management role' (engineering manager, people management, line manager, servant leadership) and the developmental subset of 'Coaching, mentoring, and career development' once 'mentor junior' is recognised as an IC-mentorship signal rather than people-mgmt authority. Recommend treating Supervisory-verbs and Team-size-signals as low-precision and requiring co-occurrence with a high-precision marker before counting a posting as people-mgmt.

_111 drops · 18 guards · 12 adds · 7 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Coaching, mentoring, and career development` — 23.9%
- `Supervisory verbs` — 11.1%
- `Team leadership` — 8.8%
- `Team-size signals` — 8.0%
- `People management role` — 2.3%
- `Performance management` — 0.9%
- `Headcount and hiring authority` — 0.5%
- `Direct reports` — 0.4%
- `1:1 meetings` — 0.2%
- `Termination, hiring/firing authority` — 0.0%

**Top guards** (false-positive risks worth fixing):
- `supervision` (Supervisory verbs): Dominantly matches 'minimal supervision', 'limited supervision', 'with little supervision', 'without supervision' — autonomy boilerplate that is the OPPOSITE of being a supervisor (the candidate works without one). → _guard_: Negative lookbehind for {minimal, minimum, little, limited, no, less, without, with little} within 4 words preceding 'supervision'. Also exclude 'under [the] supervision of'. Only count if preceded by 'provides/provide/exercises/exercise/has/with [direct]' or followed by 'of [staff|team|engineers|developers|associates|reports]'.
- `supervisory` (Supervisory verbs): Frequently matches 'no supervisory responsibilities', 'this position does not have supervisory responsibilities', 'no supervisory/management experience required' — explicit negation indicating IC role. → _guard_: Exclude any sentence containing 'no supervisory', 'does not have supervisory', 'without supervisory', 'no supervisory/management'. Require positive co-occurrence with 'responsibilities', 'experience', 'role' AND positive context.
- `supervised` (Supervisory verbs): Heavily collides with ML jargon: 'supervised learning', 'supervised and unsupervised', 'supervised fine-tuning', 'supervised learning datasets'. → _guard_: Negative lookahead for {learning, fine-tuning, model, models, dataset, datasets, training, classification, ML} within 3 words after 'supervised'. Prefer 'supervised [N] engineers/developers/staff/team' patterns.
- `supervising` (Supervisory verbs): Examples include 'supervising performance and quality assurance processes', 'supervising and updating predictive models' — supervising-things rather than supervising-people. → _guard_: Require object-level disambiguation: 'supervising {a team, [N], engineers, developers, staff, associates, the team, employees, reports, junior}'.
- `oversee` (Supervisory verbs): Dominated by technical scope: 'oversee implementations', 'oversee the cloud systems', 'oversee change control processes', 'oversee the development of our website', 'oversee the selection and integration of...technologies'. → _guard_: Require object 'oversee [the|a] {team, engineers, developers, staff, group, organization, department}' — never bare 'oversee X' where X is a system/process/project.

**Top suggested additions** (grounded in corpus snippets):
- `compensation decisions` → Performance management
- `compensation recommendations` → Performance management
- `compensation planning` → Performance management
- `career progression plans` → Coaching, mentoring, and career development
- `career progression` → Coaching, mentoring, and career development

**Concept-level recommendations:**
- `Supervisory verbs`: Concept hit rate 11.08% is the second-highest in the topic but is dominated by false positives: 'supervision' fires on 'minimal supervision' boilerplate, 'oversee/oversight' fires on technical scope, 'supervised' collides with ML jargon. Without guards, this concept overstates people-management roughly 5-10x. → _Either (a) restrict the concept to verb+human-object bigrams only ('manage/lead/supervise/oversee + the team / a team / engineers / staff / direct reports / employees') and drop the bare verb forms, or (b) split into 'Supervisory verbs (high-precision: object=people)' and 'Generic oversight verbs (low-precision)' and only use the high-precision split as a positive people-mgmt signal._
- `Team leadership`: Bundles formal team-lead phrases ('lead a team', 'manage a team') with senior-IC job titles ('tech lead', 'lead engineer', 'lead software engineer', 'technical lead', 'lead developer'). The IC-vs-manager mix is itself the dimension the paper wants to measure 2024->2026, so conflating them defeats the purpose. → _Split into two concepts: (a) 'Team leadership (formal)' containing 'lead a team', 'leading a team', 'manage a team', 'team leader', 'head of engineering', etc.; (b) 'Lead-titled roles (ambiguous IC/manager)' containing 'tech lead', 'lead engineer', 'lead developer', 'lead software engineer', 'technical lead', 'engineering lead'. Treat (b) as a separate variable, not as additive evidence for people_management._
- `Team-size signals`: Concept hit rate 8.0% is dominated by bare 'team of' (1,527 hits), which mostly captures IC team-membership ('part of a team of engineers') rather than management scope. The intended signal — quantified mgmt scope — is buried. → _Drop bare 'team of'. Replace with regex 'team of \d+' for quantified scope, plus the existing verb-prefixed forms ('manage/managing/lead/leading a team of'). Reframe the concept as 'Quantified team-size mgmt scope'._
- `Coaching, mentoring, and career development`: Concept hit rate 23.88% is by far the highest in the topic, but the dominant keyword 'mentor' (and 'mentoring', 'mentorship', 'mentor junior', 'mentor engineers') fires heavily in senior-IC contexts where the candidate is asked to mentor peers without any people-mgmt authority. As-is, this concept will mark the majority of staff/principal IC postings as people-management. → _Split into two concepts: (a) 'Informal mentorship' (mentor, mentoring, mentorship, mentor junior, mentor engineers, coaching engineers) — explicitly NOT a people-mgmt indicator on its own, OR (b) 'Career development of subordinates' (career development, career growth, professional development, professional growth, talent development, develop talent, grow engineers, develop your team, individual development plan, career progression plans). Only count (b) toward people_management; or require (a) to co-occur with a high-precision marker._
- `Termination, hiring/firing authority`: Concept hit rate is 0.0001 (1 posting in 19,433); 10/11 keywords are zero-hit. Hire/fire authority is essentially never stated explicitly in JDs, so this concept cannot serve as a positive indicator. → _Either drop the concept entirely from the people_management aggregate, or retain it as a documented near-zero baseline with a note that absence of this signal is uninformative. Do NOT treat it as a separate dimension in any factor analysis (it will be all zeros)._

### Orchestration (`orchestration`)

The classical orchestration concepts (architecture, system design, specifications, workflow/pipeline orchestration) saturate the corpus across both periods (architecture alone hits 35% of postings) and dominate hit volume, while the emerging AI-orchestration concepts (context engineering 2.6%, agent harnesses 3.9%, repo instructions 0.04%, agent memory 1.0%, multi-agent 1.0%, AI eval/steering 1.7%) are sparse but almost exclusively materialize in 2026-04 snippets. The vocabulary therefore does distinguish the 2024 vs 2026 modes well in principle, but several AI-side concepts are under-covered: many high-frequency 2026 tokens (MCP, agentic, agent orchestration, LangChain, ADK, Bedrock Agents, agentic primitives, tool-calling, agentic workflow patterns) are missing from the right concepts, and Repository Instructions / Agent Configuration Files barely fires (8 postings) because the only signals that hit are CLAUDE.md and .cursorrules, both at 1-2 hits. Several high-volume keywords are dominated by classical or unrelated meanings (Temporal=temporal data, HITL=hardware-in-the-loop, memory management=C++ heap, A2A=2024 application-to-application integration, guardrails=security/data guardrails) and need guards or AI-context co-occurrence requirements before they can carry the agentic interpretation. Net: the paper can credibly claim a 2024->2026 shift in vocabulary for context engineering, agent harnesses, and multi-agent coordination, but the Repository Instructions concept is too thin to sustain a quantitative claim and the agent-side keywords need guards to avoid contamination from classical SWE meanings.

_108 drops · 9 guards · 14 adds · 4 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Software Architecture and System Design` — 54.9%
- `Workflow Design and Pipeline Orchestration` — 39.9%
- `Specifications and Requirements Authoring` — 18.3%
- `Task Decomposition and Planning` — 12.7%
- `Agent Harnesses and Coding Agents` — 3.9%
- `Context Engineering and Prompt/Spec Authorship for Agents` — 2.6%
- `AI Tool and Agent Evaluation/Steering` — 1.7%
- `Multi-Agent Coordination` — 1.0%
- `Agent Memory and State Management` — 1.0%
- `Repository Instructions and Agent Configuration Files` — 0.0%

**Top guards** (false-positive risks worth fixing):
- `Temporal` (Workflow Design and Pipeline Orchestration): Most 2024-04 hits refer to 'temporal data', 'spatial and temporal dimensions', or 'temporal factors' in security/data-engineering contexts, NOT the Temporal.io workflow engine. Only ~20-30% of the 41 hits actually mean the orchestration tool. → _guard_: Require co-occurrence with 'workflow', 'pipeline', 'orchestration', 'durable execution', 'Temporal.io', 'Temporal Cloud', or 'pipelines with Temporal' within ~50 chars; otherwise drop the hit.
- `HITL` (AI Tool and Agent Evaluation/Steering): Almost all 28 hits, especially from 2024-01/2024-04 defense/embedded postings, mean 'Hardware-In-The-Loop' (test rigs, SITL/HITL, target platforms), not 'human-in-the-loop' for AI. Only the 2026-04 examples align with the AI-steering meaning. → _guard_: Require co-occurrence with 'AI', 'LLM', 'agent', 'model', 'human-in-the-loop', or 'review' within ~80 chars; explicitly exclude when 'Hardware in the Loop', 'SITL', 'HOOTL', 'target platform', 'flight test', or 'radar' appear in the same posting.
- `memory management` (Agent Memory and State Management): All shown 153 hits are about C++/embedded heap memory (multithreading, GDB, CPU architectures, Node.js event loop), not agent memory. The agentic meaning appears only with phrases like 'agentic memory management' or 'memory/context management for agents'. → _guard_: Require co-occurrence with 'agent', 'agentic', 'LLM', 'context', 'conversation', or 'long-term/short-term' within ~80 chars; exclude when 'multithreading', 'GDB', 'C++', 'event loop', 'embedded' appear in the same posting.
- `A2A` (Multi-Agent Coordination): 2024 hits are 'A2A/B2B integration' (application-to-application middleware/SOA), not Agent-to-Agent protocol. 2026-04 hits are dominantly the agentic meaning (paired with MCP, AutoGen, LangChain). → _guard_: Require co-occurrence with 'agent', 'MCP', 'protocol', 'multi-agent', 'AutoGen', 'LangChain', 'CrewAI', or 'agentic' within ~80 chars; exclude when 'B2B' or 'SOA' is in the same snippet.
- `guardrails` (AI Tool and Agent Evaluation/Steering): Of 205 hits, the majority (especially 2024) refer to security guardrails, data quality guardrails, deployment guardrails — not AI/agent guardrails. Only when paired with 'GenAI', 'LLM', 'agent', 'AI' is it the right concept. → _guard_: Require co-occurrence with 'AI', 'LLM', 'GenAI', 'agent', 'model', or 'prompt' within ~80 chars; otherwise drop. ('AI guardrails' as a separate keyword already does this and should be retained.)

**Top suggested additions** (grounded in corpus snippets):
- `MCP` → Agent Harnesses and Coding Agents
- `Model Context Protocol` → Agent Harnesses and Coding Agents
- `agentic` → Agent Harnesses and Coding Agents
- `LangChain` → Agent Harnesses and Coding Agents
- `LlamaIndex` → Agent Harnesses and Coding Agents

**Concept-level recommendations:**
- `Repository Instructions and Agent Configuration Files`: Concept fires in only 8 of 19,433 postings (0.04%) and 18 of 23 keywords are zero-hit. The repo-instructions ecosystem (CLAUDE.md, AGENTS.md, .cursorrules, copilot-instructions.md) is essentially absent from job ads — practitioners apparently don't list these in postings even when they use them. As designed, this concept will not support a quantitative claim and risks being misread as 'agent configuration is not a thing' in 2026. → _Either (a) merge this concept into 'Context Engineering and Prompt/Spec Authorship for Agents' since the underlying skill is the same (writing structured guidance for agents) and the merged concept then reaches ~510 postings; or (b) keep but reframe as a *qualitative* concept and document in the paper that absence here is itself a finding (mainstream postings haven't yet adopted file-level repo-instruction vocabulary). Do not delete the few keywords that did fire (CLAUDE.md, .cursorrules, agent configuration) — they remain useful as canaries._
- `Agent Memory and State Management`: Concept fires in 186 postings but 'memory management' alone (which dominates) is essentially all C++ heap management — without a guard the concept's signal is ~80% noise from classical embedded postings. → _Add the guard for 'memory management' (see guards section), promote 'agentic memory', 'agent memory', 'long-term memory' (in agent context), and consider adding 'conversation state', 'session state', 'agent state', 'checkpointing' (last is in LangGraph snippet) as the load-bearing keywords. Without a guard the concept will read as tens of times larger than it really is._
- `AI Tool and Agent Evaluation/Steering`: Two of the three biggest contributors ('guardrails' 205, 'HITL' 28) are dominated by non-AI meanings (security guardrails, hardware-in-the-loop). Without guards the concept hit-rate of 1.7% is overstated by a factor of ~5-10x. → _Apply guards on 'guardrails' and 'HITL'; promote 'human-in-the-loop' (which is already mostly correct), 'AI guardrails' (16, all 2026), 'evals', 'LLM evals', 'agent evaluation' as primary signals. Consider adding 'review AI-generated code' variants and 'tracing' / 'observability' if cross-list rules permit._
- `Multi-Agent Coordination`: Half the keywords are zero-hit and the concept hits only 201 postings, but the load-bearing keywords ('multi-agent', 'A2A', 'multi-agent orchestration') ARE all 2026-heavy. The concept is correctly aimed; it just needs cleanup of speculative variants (agent swarm, agent supervisor, planner agent, worker agent — none of which fire) and the additions (agent orchestration, supervisor-worker pattern) noted above. → _Drop the zero-hit speculative variants, add 'agent orchestration' and 'supervisor-worker' as new keywords, guard 'A2A' for 2024 application-to-application contamination._

### Verification (`verification`)

The verification vocabulary is generally well-targeted: CI/CD, observability, code review, and automated-testing concepts each fire on 10-40% of postings, dominated by unambiguous compound terms (CI/CD, code reviews, monitoring, unit testing). The biggest false-positive risk is the QA-and-validation concept: bare 'validation/validate/verify/verification' frequently capture systems-engineering V&V boilerplate, requirements-validation, identity-verification, and even 'E-Verified Company' tag-lines rather than software correctness gating. The Evaluations concept is anemic at hit_rate=0.097 because its core terms ('eval', 'evals', 'LLM eval(s)', 'eval suite/harness') barely fire on a 2024-skewed sample, while 'evaluation/evaluations' over-fire on benefits/performance reviews and generic 'evaluation of third-party libraries.' 75/432 keywords are zero-hit (mostly speculative AI-eval and CI-gate variants); recommended adds focus on the 2026 AI-eval stack actually visible in scraped snippets (LangSmith, Promptfoo, OpenAI Evals, Braintrust). 'Black' (under static analysis) is dominated by 'black-box testing' and 'Black Duck' rather than the Python formatter and should be dropped or guarded.

_84 drops · 22 guards · 14 adds · 4 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `CI/CD and quality gates` — 40.8%
- `Post-deployment observability` — 34.2%
- `QA and validation` — 29.2%
- `Automated testing` — 22.8%
- `Code review` — 17.9%
- `Compliance and security verification` — 16.8%
- `Static analysis and code quality tooling` — 15.5%
- `Reproducibility and artifact proof` — 10.3%
- `Evaluations and AI output verification` — 9.7%

**Top guards** (false-positive risks worth fixing):
- `QA` (QA and validation): Genuinely a QA-function reference in 95% of snippets, but two-letter acronym risks Q&A confusion. Spot-check shows clean usage in this corpus. → _guard_: Require uppercase 'QA' as token (case-sensitive); reject 'Q&A' and 'Q & A' contexts. Optionally require co-occurrence with 'test|quality|engineer|tester' within 50 chars.
- `validation` (QA and validation): Heavy noise from systems-engineering V&V ('verification and validation', 'validation of customer needs'), data-science 'cross-validation', identity 'validation', and form/input validation in app code. → _guard_: Require co-occurrence with 'test|QA|automated|software|code|build|CI' within 80-char window, or require collocation 'data validation|input validation|unit validation' to be excluded. Consider only counting validation when paired with test/automation context.
- `validate` (QA and validation): Often generic ('validate assumptions', 'validate model effectiveness', 'validate customer requirements') — not software correctness gating. → _guard_: Same as 'validation'; require software/test co-occurrence.
- `validating` (QA and validation): Same generic-validation FP pattern. → _guard_: Require software-test context.
- `validated` (QA and validation): Mostly generic. → _guard_: Require software-test context.

**Top suggested additions** (grounded in corpus snippets):
- `LangSmith` → Evaluations and AI output verification
- `LangFuse` → Evaluations and AI output verification
- `Promptfoo` → Evaluations and AI output verification
- `OpenAI Evals` → Evaluations and AI output verification
- `Braintrust` → Evaluations and AI output verification

**Concept-level recommendations:**
- `Reproducibility and artifact proof`: Concept is 41 keywords with hit_rate 0.103, but the bulk of hits come from 'log', 'logs', 'logging', 'trace', 'traces', 'tracing', 'artifact', 'artifacts' — which are pure observability or build-pipeline terms, not reproducibility/proof. The 'reproducibility' core (reproducible/reproduce/formal verification/proof of correctness) collectively fires on <2% of postings. The concept conflates two unrelated ideas. → _Split: (1) move 'log/logs/logging/trace/traces/tracing' to Post-deployment observability; (2) move 'artifact/artifacts/build artifact[s]/audit trail/audit log' to a 'CI/CD artifacts' sub-concept under CI/CD; (3) keep a slim 'Formal verification and reproducibility' concept around 'reproducibility|reproducible|formal verification' — accept that this concept will have <2% hit-rate, and that's the honest finding._
- `Evaluations and AI output verification`: Concept hit-rate of 0.097 is inflated by generic 'evaluation' (1327 hits) and 'evaluations' (189 hits), the majority of which are non-AI ('performance evaluations for assigned staff', 'field evaluations', 'evaluation of third-party libraries'). After guarding those down, the concept's true AI-eval hit-rate is likely <3%, dominated by 'guardrails' (n=205) and 'evaluation framework[s]' (n=109). → _Apply AI/ML/model/LLM co-occurrence guard to 'evaluation/evaluations'. Add 2026-vintage tooling (LangSmith, LangFuse, Promptfoo, OpenAI Evals, Braintrust, Helicone, Arize). Expect a sharp 2024->2026 increase that is the headline finding for this topic, not a steady-state hit rate._
- `Static analysis and code quality tooling`: 'TypeScript' (1847 hits) is bundled here as type-safety signal, but TypeScript-as-language overwhelmingly dominates (it's a stack mention, not a verification signal). 'Black' (75 hits) is almost entirely 'black-box testing' and 'Black Duck'. → _Drop 'TypeScript' from this concept (it's a language, not a verification practice — keep elsewhere if relevant). Drop 'Black' entirely. Type-safety is better captured by a guarded 'mypy|Pyright|tsc|strict mode' set, which the current zero-hit list shows is too sparse to support a sub-concept._
- `Code review`: 'design review' (n=72) and 'design reviews' (n=621) are largely architecture/system design reviews and even 'electrical design review of new products' — not code review. → _Move 'design review[s]' to a separate 'Architecture/design review' sub-concept under collaboration or design, OR exclude entirely from Code review. Mentorship cross-list collisions on 'code review[s]' / 'review code' are claim-overlapping (a posting that mentions code reviews legitimately supports both verification-via-review and mentorship-via-review); flag these examples for the verification claim only when the surrounding text emphasizes correctness/quality (e.g., 'code review process to catch defects') rather than teaching._

### Mentorship markers (`mentorship`)

The mentorship vocabulary is large (474 keywords across 11 concepts) but heavy with zero/low-hit phrasal variants (141 zero-hits, ~30%) and a handful of high-volume bare verbs that grossly inflate hit rates with off-topic matches. The 'Guidance and influence' concept (29.1% hit rate) and 'Code review as teaching' concept (19.2%) are inflated by terms that fire mostly on non-mentorship usage: bare 'guide/guides/guidance' overwhelmingly captures user guides, missile guidance, and process guidance, not engineer guidance; bare 'code review/code reviews/design reviews/peer reviews' fire on quality-gate code review (a verification activity), not on review-as-teaching. Similarly, 'Debugging and code-reading as a craft' (17.6%) is essentially driven by the bare verb 'debug/debugging' which fires on every job posting that mentions writing/maintaining code, with no mentorship signal. The 'Explicit learning culture' concept also sweeps in self-directed-learning vocabulary (curious, curiosity, continuous learning, growth mindset, professional development) that is about the candidate's own growth, not about teaching others. Strong, well-targeted concepts are 'Direct mentorship language', 'Pair / mob programming' (small but precise), 'Onboarding', and the 2026 scraped corpus shows novel, high-signal patterns ('level them up', 'raise the bar', pair-programming with AI tools like Devin/Cline) that are not in the current list.

_203 drops · 90 guards · 12 adds · 7 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Guidance and influence` — 29.1%
- `Direct mentorship language` — 21.8%
- `Code review as teaching` — 19.2%
- `Debugging and code-reading as a craft` — 17.6%
- `Explicit learning culture` — 7.8%
- `Teaching and knowledge transfer` — 6.7%
- `Architecture and design guidance` — 6.4%
- `Coaching and developing engineers` — 4.6%
- `Onboarding and ramp-up` — 2.9%
- `Pair / mob / ensemble programming` — 1.5%
- `Communication and feedback skills` — 1.2%

**Top guards** (false-positive risks worth fixing):
- `guide` (Guidance and influence): Bare 'guide' fires on user guides, style guides, activity guides, FDA-guided environments, missile guidance, and 'guide a team' (people management). Only ~10-20% of 1111 hits look like engineer-guidance. → _guard_: Require an engineering object: 'guide (?:engineers|developers|the team|junior|peers|teammates|other engineers|other developers|colleagues)' OR collocated with 'mentor'/'coach' within ~30 chars. Drop bare 'guide' as a standalone trigger.
- `guides` (Guidance and influence): Mostly Activity Guides, user training guides, written documents (noun plural). → _guard_: Drop bare 'guides' or restrict to 'guides (?:engineers|developers|the team|junior|peers)'.
- `guided` (Guidance and influence): FDA-guided, guided by culture/philosophy, missile guidance navigation. → _guard_: Drop unless preceded by a person/role: '(?:has|is) guided (?:engineers|developers|teams|juniors)'.
- `guiding` (Guidance and influence): 'guiding projects', 'guiding teams toward secure solutions' is borderline; many hits fire on 'guiding principles' or 'guiding documents'. → _guard_: Require people object within window: 'guiding (?:engineers|developers|junior|teammates|peers|the team|other engineers)'.
- `guidance` (Guidance and influence): Hugely inflated by 'guidance, navigation, and control' (missile/aerospace), 'state enterprise architecture guidance', 'compliance guidance', 'governance guidance', 'guidance documents'. → _guard_: Require modifier: '(?:technical|engineering|architectural|architecture|design|hands-on) guidance' OR co-occurrence with 'mentor|coach|junior|develop|teach' in same sentence.

**Top suggested additions** (grounded in corpus snippets):
- `level them up` → Coaching and developing engineers
- `level up the team` → Coaching and developing engineers
- `raise the bar for` → Coaching and developing engineers
- `high-feedback culture` → Communication and feedback skills
- `high-trust` → Explicit learning culture

**Concept-level recommendations:**
- `Code review as teaching`: The bare 'code review/code reviews/peer review/peer reviews/design review/design reviews' keywords (totaling thousands of hits) capture the verification activity of code review, NOT code review framed as teaching. This concept will inflate mentorship hit-rate by sweeping in nearly every SWE posting (since virtually all describe code review as a basic practice). The concept's hit rate of 19.2% is misleading. → _Either (a) remove bare review terms and rely only on teaching-framed variants ('thoughtful code review', 'constructive code review', 'review that teaches', 'design discussions', 'pairing-driven review'), with co-occurrence guard requiring mentor/teach/grow within ~50 chars; or (b) split into two sub-concepts and explicitly mark bare review terms as a 'verification overlap' that should be subtracted from the mentorship signal during reconciliation._
- `Debugging and code-reading as a craft`: Bare 'debug/debugs/debugging/debugged' (combined ~3400 hits driving the 17.6% concept hit-rate) are core SWE activity terms, not mentorship signals. They fire on every 'design, develop, test, debug' boilerplate. The concept's framing — debugging as a learning/teaching activity — is not actually captured by these tokens. → _Drop bare debug verbs entirely. Reframe concept around explicitly-pair/teach-context terms only: 'pair debugging', 'walk through code', 'code walkthroughs', 'paired debugging'. If those produce too few hits to support the concept, consider folding the survivors into 'Pair / mob / ensemble programming' and dissolving this concept._
- `Guidance and influence`: This concept conflates four distinct things: (a) mentorship-by-influence ('mentor and guide', 'role model'), (b) people-management/title vocabulary ('tech lead', 'engineering leadership', 'technical leadership'), (c) external/customer-facing thought-leadership ('thought leader', 'advisor', 'consulting'), and (d) generic noun 'guidance/guide' that fires on user guides and missile guidance. Concept hit-rate (29.1%) is therefore meaningless. → _Split into: (1) 'Mentorship through influence' (lead by example, leading by example, lead through influence, influence without authority, role model, mentor and guide, shepherd ENGINEERS); (2) 'Senior-IC titles and advisory positioning' (cross-list this with people_management); (3) drop bare 'guide', 'guides', 'guidance', 'consulting', 'consult', 'advise/advisor' OR force engineering-target guards as listed above. The current collisions with people_management (already tagged on 'mentor', 'mentors', 'mentored', 'coach', 'coaches', 'coached') confirm boundary-drift._
- `Explicit learning culture`: Concept mixes two semantically distinct things: (a) candidate self-growth disposition ('curious', 'curiosity', 'growth mindset', 'passion for learning', 'lifelong learner', 'continuous learning') and (b) team-level learning culture that the role is expected to foster ('culture of learning', 'foster growth', 'invest in career development of your employees'). Mentorship is fundamentally about (b), not (a). Most current keyword hits in this concept are (a). → _Restrict the concept to team-directed learning culture only: drop 'curious/curiosity/intellectual curiosity/intellectually curious/growth mindset/passion for learning/love to learn/lifelong learner/lifelong learning/continuously learn/always learning'. Keep 'learning culture', 'culture of learning', 'foster growth', 'support the growth', and pair them with 'team/engineers/others' guards. The candidate-disposition vocabulary likely belongs to a separate 'candidate disposition / soft-skill' topic, not mentorship._
- `Architecture and design guidance`: The 'best practices' and 'design documents' subset overlaps heavily with architecture/verification topics rather than mentorship. 'Establish/define/drive/promote/evangelize best practices' is architecture standards-setting; only with mentor/junior/teach co-occurrence does it become a mentorship act. 'Design documents/design docs' are pure architecture artifacts. → _Move the bare 'best practices' phrasing and 'design documents/design docs' to an architecture concept; keep here only when co-occurring with mentor/junior/team. The genuine mentorship signal in this concept is 'architectural guidance', 'design guidance', 'design mentorship', 'raise the technical/engineering bar' — keep those._

### Performance & deep technical understanding (`performance`)

The list is structurally sound for technical perf signals (profiling, latency/throughput, low-level systems, hardware-aware) where keywords are sufficiently specific and corroborated by 2026 snippets (sub-millisecond, p99, ultra-low-latency, performance-critical, performance regression, microbenchmarks, ftrace, distributed tracing). The depth_claim concept, however, is heavily contaminated by HR/recruiter fluff: 'expert level' (often a literal section header, no technical noun), 'expert in' / 'expert knowledge' (frequently bound to non-perf nouns like Angular, Mainframe), 'mastery' (Python/JavaScript), 'fluent in' (HTML/CSS/SQL), 'fundamentals' (generic 'web development fundamentals'), and 'from scratch'/'ground up' (generic build framing). Without applying the technical-noun proximity guard from the exclusions section, the list will overstate depth-claim language and bias the time trend if recruiter fluff is rising independently of technical depth. With the proximity guards enforced, the list CAN measure depth-claim trend reliably; without them, the signal is dominated by HR language and unsuited for the hypothesis test. Additionally 33% of keywords (191/582) are zero-hits (mostly British spellings, micro-jargon like 'L1 cache', 'IRQ', 'page table', 'B-tree internals') — recommend pruning to keep the vocab readable, but they don't bias the measurement.

_80 drops · 28 guards · 14 adds · 4 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `depth_claim_language` — 38.9%
- `scaling_and_efficiency` — 23.2%
- `latency_throughput_scale` — 14.6%
- `performance_optimization_general` — 13.9%
- `low_level_systems_programming` — 12.7%
- `profiling_and_benchmarking` — 11.7%
- `distributed_systems_internals` — 8.7%
- `algorithmic_optimization` — 5.4%
- `concurrency_parallelism` — 5.1%
- `hardware_aware_programming` — 2.9%
- `compilers_and_runtimes` — 1.6%
- `os_kernel_internals` — 1.6%
- `database_storage_internals` — 1.0%
- `networking_internals` — 0.7%

**Top guards** (false-positive risks worth fixing):
- `high performance` (performance_optimization_general):  → _guard_: 
- `high-performance` (performance_optimization_general):  → _guard_: 
- `high performance computing` (performance_optimization_general):  → _guard_: 
- `expert level` (depth_claim_language):  → _guard_: 
- `expert-level` (depth_claim_language):  → _guard_: 

**Top suggested additions** (grounded in corpus snippets):
- `?` → performance_optimization_general
- `?` → performance_optimization_general
- `?` → profiling_and_benchmarking
- `?` → profiling_and_benchmarking
- `?` → compilers_and_runtimes

**Concept-level recommendations:**
- `depth_claim_language`: Currently mixes three distinct signals: (1) genuine depth-claim language ('deep understanding of X', 'first principles', 'intimate knowledge of', 'mastery of'), (2) recruiter section headers / level markers ('expert level' as standalone), (3) tool-fluency framing ('fluent in JS', 'fluency in SQL', 'mastery of Python'), and (4) greenfield/construction framing ('from scratch', 'from the ground up'). For RQ-relevant trend measurement, signals (1) and (2)/(3)/(4) likely move on different time-axes (HR fluff inflation vs technical-depth shift). → __
- `performance_optimization_general`: Conflates technical perf ('high-performance algorithms', 'performance-critical', 'performance bottlenecks', 'performance regression') with HR fluff ('high performance team/culture/environment') under the same 'high performance' / 'high-performance' keywords. → __
- `concurrency_parallelism`: Two keywords ('channels', 'fiber') are essentially false-positive only in this corpus — they map to marketing/storage/optical contexts, not concurrency. They inflate the concept hit_rate without providing real concurrency signal. → __
- `scaling_and_efficiency`: 'efficiency' (2033 hits, hr=0.1046) and 'scalability' (2213 hits, hr=0.1139) are extremely generic JD terms not specific to performance — 'workflow efficiency', 'deployment efficiency', 'efficient delivery' are in the 2024 examples. These two keywords single-handedly drive the concept hit_rate (0.2316) and likely measure JD-length / generic framing more than technical perf. → __

### Process-scaffolding markers (`process_scaffolding`)

The list is functionally usable but heavily contaminated by section-header and acronym false positives that will swamp any 2024->2026 trend signal unless guarded. The single most damaging issue is bare 'requirements' (49% hit rate, 9514/19433 postings) — examples confirm it overwhelmingly fires on the boilerplate 'Requirements:' JD section header, not on requirements engineering activity; the 'Requirements engineering' concept's 53% concept-level hit rate is essentially noise. Several short acronyms catch unrelated meanings at high volume and must be guarded or dropped: 'DoD' (727 hits, almost all Department of Defense clearances), 'ART' (515 hits, mostly 'state-of-the-art'), 'LeSS' (567 hits, mostly the CSS preprocessor or the word 'less'), 'PLC' (110, programmable logic controller), 'Linear' (141, linear algebra/regression), 'PM' (146, post-meridiem and trader desks), 'PO' (purchase order/SAP), 'SM', 'CSM' (ServiceNow Customer Service Management), 'Epic' (the EHR vendor), 'XP' (Windows XP), 'CR', 'CAB', 'TPM' (Trusted Platform Module), 'V model' (vehicle data/OSI model), 'NFR' (a company name), and 'ERD' (Entity Relationship Diagram, not Engineering Requirements Document). Likely-net-positive but with substantial collision: 'validation'/'verify'/'verify' (already cross-listed with the verification topic), 'governance' (data/cloud governance), 'lean' (LEAN manufacturing, 'lean towards'), 'velocity' (developer velocity), 'iterative' (general adjective), 'roadmap' (technology roadmap, also product). 62 zero-hit and 22 low-hit-likely-noise keywords can be dropped outright with no information loss. Net: the process-scaffolding signal is real (Agile, Scrum, Jira, SDLC, sprint planning, ceremonies all hit cleanly at sensible rates), but if the paper reports the raw hit rate of this list 2024->2026 it will be dominated by 'requirements:' headers and 'DoD clearance' which have nothing to do with process scaffolding. Recommend (a) replacing bare 'requirements' / 'requirement' with phrase forms only ('requirements gathering', 'requirements analysis', 'business requirements', etc.), (b) guarding all acronym tokens with disambiguation context, and (c) re-running the headline metric on the cleaned list before drawing conclusions about the methodology layer shrinking.

_127 drops · 40 guards · 4 adds · 4 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Requirements engineering` — 53.1%
- `SDLC and process governance` — 34.2%
- `Agile methodology` — 31.2%
- `Scheduling and milestones` — 23.0%
- `Project / program management roles & tooling` — 19.0%
- `Verification & Validation (V&V)` — 16.4%
- `Project coordination` — 15.8%
- `Scrum framework` — 13.4%
- `Specification authoring` — 12.8%
- `Sprints and iterations` — 9.5%
- `Kanban / Lean / Waterfall and other methodologies` — 9.3%

**Top guards** (false-positive risks worth fixing):
- `requirements` (Requirements engineering):  → _guard_: 
- `requirement` (Requirements engineering):  → _guard_: 
- `DoD` (Requirements engineering):  → _guard_: 
- `ART` (SDLC and process governance):  → _guard_: 
- `LeSS` (Kanban / Lean / Waterfall and other methodologies):  → _guard_: 

**Top suggested additions** (grounded in corpus snippets):
- `?` → Agile methodology
- `?` → Sprints and iterations
- `?` → SDLC and process governance
- `?` → Scheduling and milestones

**Concept-level recommendations:**
- `Requirements engineering`: Concept hit rate 53.1% is artifactually high. Bare 'requirements' (9514 hits) and 'requirement' (945) are catching the JD section header 'Requirements:' overwhelmingly. After dropping/guarding the bare tokens, expect concept hit rate to fall to ~10-15% reflecting actual requirements-engineering activity (gather/elicit/translate/document requirements + business/system/functional requirements + use cases + acceptance criteria + DoD). RECOMMENDATION: redefine concept with phrase-only matching (no bare 'requirement(s)'), or apply a header-detection pre-filter that strips 'Requirements:' / 'Job Requirements:' / 'Minimum Requirements:' section headers before keyword matching. → __
- `Verification & Validation (V&V)`: Concept hit rate 16.4% is dominated by bare verbs validate/verify/validation/verification (all 4 cross-list-collide with the verification topic). Process-level V&V (V&V, IV&V, validation plans, qualification testing, design verification) is the intended signal but quantitatively dwarfed by generic test-level 'data validation', 'input validation' etc. Either reconcile cross-list (out of scope here) or note that bare verbs measure QA-testing prevalence not process-V&V. → __
- `SDLC and process governance`: 'governance' (1345 hits) is mostly data/cloud/AI governance, not SDLC governance. 'ART' (515) is mostly 'state-of-the-art'. After guarding/dropping these, this concept's true rate is closer to the SDLC family (15-20%) which is the intended signal. → __
- `Project / program management roles & tooling`: Several acronym tokens (PM, TPM, TPMs, Linear) catch unrelated meanings dominantly. The Jira/Confluence/Atlassian/JIRA group is a clean signal of process-tooling presence and should be the primary indicator for this concept; PM-role tokens need guards or removal. → __

### Legacy-stack markers (`legacy_stack`)

The list cannot reliably identify legacy-stack JDs in its current form: roughly a third of the 300 keywords are zero-hit (precise version strings like 'EJB 2', 'Exchange 2010', 'IIS 6', 'Oracle 9i', 'on-prem AD' that nobody writes that way), and several high-volume entries are catastrophic false positives - 'Make' fires on 2,218 postings (11.4% of the sample) almost entirely as the verb, and 'World' fires on 1,606 postings (8.3%) for 'world-class' / 'real-world' / company-name 'JD Edwards World'. Together those two keywords would dominate any density score and produce a wildly inflated 'legacy' signal in modern JDs. Among the truly ambiguous tech tokens, 'SOAP' (582 hits) and 'SOA' (234 hits) appear overwhelmingly alongside REST/microservices/cloud and are not legacy markers in 2024-2026 usage; '.NET Framework' (247 hits) does generally indicate the 4.x runtime but co-occurs with .NET Core in ~half of snippets so it is a soft signal at best; bare 'Active Directory' is dominated by Azure AD / Entra / hybrid-cloud contexts, not on-prem-only AD. Strong, low-noise signals do exist (COBOL, JCL, CICS, VSAM, IMS, z/OS, J2EE, WebSphere, WebLogic, JSP, EJB, ClearCase, Classic ASP, ASP.NET WebForms, VB6, Delphi, PowerBuilder, ColdFusion, OpenVMS, BizTalk, AS/400, iSeries, Sybase, Informix, ABAP, PeopleSoft, JD Edwards, Siebel, Documentum, FileNet, Hyperion, Tivoli, SiteMinder, MQSeries/IBM MQ, Oracle Forms, DB2 (with caveats)) - the topic should be redefined as the union of these strong signals plus co-occurrence-gated weak signals, and many bare/ambiguous tokens should be dropped or rewritten as multi-word patterns. Reliability is currently fixable but only after aggressive guards.

_79 drops · 29 guards · 8 adds · 8 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Legacy version control and build` — 12.6%
- `Legacy enterprise applications and ERPs` — 8.9%
- `Legacy Java EE / app server stack` — 3.6%
- `Legacy database platforms` — 3.4%
- `Legacy enterprise SOAP / web services` — 3.1%
- `Legacy enterprise integration / ESB / messaging` — 2.7%
- `Legacy Microsoft .NET stack` — 2.4%
- `Legacy Microsoft server / collaboration` — 2.3%
- `Legacy virtualization and on-prem infrastructure` — 2.0%
- `Legacy identity and directory services` — 1.8%
- `Mainframe languages and platforms` — 1.4%
- `Legacy general-purpose languages` — 0.6%

**Top guards** (false-positive risks worth fixing):
- `.NET Framework` (?):  → _guard_: 
- `dotnet framework` (?):  → _guard_: 
- `Active Directory` (?):  → _guard_: 
- `SOAP` (?):  → _guard_: 
- `SOAP web services` (?):  → _guard_: 

**Top suggested additions** (grounded in corpus snippets):
- `?` → ?
- `?` → ?
- `?` → ?
- `?` → ?
- `?` → ?

**Concept-level recommendations:**
- `TOPIC-LEVEL: split into 'definitely legacy' vs 'ambiguous (legacy or modern)' tiers`:  → __
- `Legacy Microsoft .NET stack`:  → __
- `Legacy enterprise SOAP / web services`:  → __
- `Legacy virtualization and on-prem infrastructure`:  → __
- `Legacy identity and directory services`:  → __

### Context infrastructure (`context_infrastructure`)

The list cannot cleanly distinguish 'real context-infrastructure work' from generic HR-fluff communication boilerplate without aggressive guarding. Three of the eleven concepts are saturated by JD filler: Cross-functional communication & coordination (26.7%, dominated by 'cross-functional' alone at 17.8% and 'cross-functional teams' at 11.5%), Product & business literacy (21.1%, half driven by 'business requirements'/'business needs' which appear as one-word translation cliches), and Technical writing craft (9.9%, where 'written communication' alone hits 1,123 postings — almost always the 'excellent verbal and written communication skills' cliche, not actual technical writing). Observability & telemetry (34.2%) and data-pipeline hygiene (20.7%) are credible because they fire on named tools (Datadog, Grafana, Splunk, Snowflake, Airflow, dbt). The high-signal concepts (Runbooks, ADRs/RFCs, API docs, system understanding) all fire below 7% and contain real artifact language. Recommendation: keep the artifact-side concepts as primary signal; demote written-communication and cross-functional-collaboration concepts to a separate 'soft-skill saturation' axis or report only their tail-end specific phrases. Also tighten the 'documentation' family — bare 'design documentation' / 'system documentation' / 'engineering documentation' / 'comprehensive documentation' fire heavily on non-SWE roles (PLC programmers, mainframe devs, CAD work, automation engineers) and on Salesforce/ServiceNow boilerplate, so they need either a SWE-role gate or co-occurrence with code/repo/system terms.

_101 drops · 32 guards · 8 adds · 4 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Observability & telemetry stack` — 34.2%
- `Cross-functional communication & coordination` — 26.7%
- `Product & business literacy` — 21.1%
- `Data-pipeline & data-integration hygiene` — 20.7%
- `Technical documentation authoring & maintenance` — 10.9%
- `Technical writing craft` — 9.9%
- `Architecture decision records & RFCs` — 6.5%
- `System understanding & internal knowledge sharing` — 3.6%
- `Runbooks, playbooks, & operational docs` — 3.1%
- `API & interface documentation` — 2.7%
- `Service-level reliability targets` — 2.4%

**Top guards** (false-positive risks worth fixing):
- `documentation (bare)` (Technical documentation authoring & maintenance):  → _guard_: 
- `technical documentation` (Technical documentation authoring & maintenance):  → _guard_: 
- `design documentation` (Technical documentation authoring & maintenance):  → _guard_: 
- `system documentation` (Technical documentation authoring & maintenance):  → _guard_: 
- `comprehensive documentation` (Technical documentation authoring & maintenance):  → _guard_: 

**Top suggested additions** (grounded in corpus snippets):
- `?` → Technical documentation authoring & maintenance
- `?` → Technical documentation authoring & maintenance
- `?` → API & interface documentation
- `?` → Architecture decision records & RFCs
- `?` → Technical documentation authoring & maintenance

**Concept-level recommendations:**
- `Cross-functional communication & coordination`: Concept fires on 26.7% of postings, dominated by 'cross-functional' (17.8%) and 'cross-functional teams' (11.5%) — pure HR boilerplate. The concept does not measure 'context-infrastructure work'; it measures whether the JD was authored by a recruiter who used the standard template. → __
- `Technical writing craft`: Concept appears moderate at 9.9%, but 1,123 of the 1,916 hits come from 'written communication' alone — which is the 'excellent verbal and written communication skills' HR cliche, not technical writing. Removing or guarding 'written communication' drops the concept to roughly 4-5%. → __
- `Product & business literacy`: Concept fires on 21.1% of postings, with 'business requirements' (7.1%) and 'business needs' (5.5%) dominating. These overwhelmingly appear as 'translate business requirements/needs into technical solutions' — a one-line cliche that says nothing about depth of product/business literacy. → __
- `Technical documentation authoring & maintenance`: 10.9% headline hit rate is reasonable, but the dominant terms ('technical documentation' 665, 'design documentation' 166, 'system documentation' 137, 'comprehensive documentation' 101, 'clear documentation' 80) include heavy non-SWE contamination (PLC programming, CAD/Visio, mainframe COBOL, Salesforce/ServiceNow boilerplate, automation/firmware engineering). These postings entered the SWE sample but the documentation language is generic engineering boilerplate. → __

## Cross-list reconciliation

`collisions.json` lists every keyword appearing in ≥2 topics. The patterns below cluster the most common collisions and propose a canonical home for each.

**Top topic-pair collision counts:**

| Topic A | Topic B | # collisions |
|---|---|---:|
| context_infrastructure | verification | 46 |
| orchestration | process_scaffolding | 28 |
| context_infrastructure | process_scaffolding | 25 |
| mentorship | people_management | 24 |
| context_infrastructure | orchestration | 20 |
| context_infrastructure | mentorship | 18 |
| mentorship | verification | 15 |
| process_scaffolding | verification | 13 |
| performance | verification | 13 |
| orchestration | verification | 9 |

**Proposed reconciliation rules:**

### `design_artifacts_cluster`
- **Pattern:** ADR / PRD / design doc / spec doc as written artifacts
- **Examples:** `adr`, `architecture decision record`, `prd`, `product requirements document`, `design doc`, `design docs`, `design documents`, `tech spec`, `rfc`
- **Currently in:** orchestration, process_scaffolding, context_infrastructure
- **Canonical home:** `context_infrastructure`
- **Rationale:** Artifacts are the substrate; orchestration is about authoring the work, process_scaffolding is about governance. The artifact noun lives with the substrate.
- **Action:** drop these keywords from orchestration, process_scaffolding, keep in `context_infrastructure`. (Or, for a *specific phrase* that genuinely captures the alias topic — e.g., 'mentor through code review' — keep narrowly.)

### `spec_authoring_activity`
- **Pattern:** Writing / decomposing / authoring specs as an activity
- **Examples:** `write specs`, `author specifications`, `decompose requirements`, `task decomposition`, `scope decomposition`
- **Currently in:** orchestration, process_scaffolding
- **Canonical home:** `orchestration`
- **Rationale:** Authoring activity for AI/agent consumption is the new senior archetype the paper tracks; classical SDLC verbs ('requirements gathering', 'change request') stay in process_scaffolding.
- **Action:** drop these keywords from process_scaffolding, keep in `orchestration`. (Or, for a *specific phrase* that genuinely captures the alias topic — e.g., 'mentor through code review' — keep narrowly.)

### `code_review`
- **Pattern:** Code review
- **Examples:** `code review`, `code reviews`, `pr review`, `pull request review`, `review pull requests`
- **Currently in:** verification, mentorship
- **Canonical home:** `verification`
- **Rationale:** Code review's primary JD framing is correctness gating. 'Mentor through code review' / 'use reviews to teach' stays in mentorship as a more specific phrase, but bare 'code review' belongs in verification.
- **Action:** drop these keywords from mentorship, keep in `verification`. (Or, for a *specific phrase* that genuinely captures the alias topic — e.g., 'mentor through code review' — keep narrowly.)

### `design_review`
- **Pattern:** Design review
- **Examples:** `design review`, `design reviews`, `architecture review`, `architecture reviews`
- **Currently in:** verification, mentorship, context_infrastructure
- **Canonical home:** `context_infrastructure`
- **Rationale:** Design/architecture review is most often framed as a governance/quality artifact-process. Mentorship version captures 'teach through architecture review' — keep there as specific phrase only.
- **Action:** drop these keywords from verification, mentorship, keep in `context_infrastructure`. (Or, for a *specific phrase* that genuinely captures the alias topic — e.g., 'mentor through code review' — keep narrowly.)

### `observability`
- **Pattern:** Observability
- **Examples:** `observability`, `telemetry`, `metrics, logs, traces`, `distributed tracing`, `monitoring`
- **Currently in:** verification, performance, context_infrastructure
- **Canonical home:** `context_infrastructure`
- **Rationale:** Observability as a substrate (dashboards, telemetry hygiene) lives in context_infrastructure. Verification claims 'post-deployment observability for catching regressions' — a specific framing, keep narrowly. Performance claims 'profiling/perf telemetry' — a specific framing, keep narrowly.
- **Action:** drop these keywords from verification, performance, keep in `context_infrastructure`. (Or, for a *specific phrase* that genuinely captures the alias topic — e.g., 'mentor through code review' — keep narrowly.)

### `leadership_vs_management`
- **Pattern:** Lead/leadership/principal language
- **Examples:** `technical lead`, `tech lead`, `lead engineer`, `principal engineer`, `staff engineer`
- **Currently in:** mentorship, people_management
- **Canonical home:** `people_management`
- **Rationale:** These are role/seniority titles, not mentorship verbs. Mentorship should keep the verbs ('mentor junior engineers', 'grow the team') and yield the title nouns to people_management — but note that 'lead' is highly polysemous and needs a separate guard.
- **Action:** drop these keywords from mentorship, keep in `people_management`. (Or, for a *specific phrase* that genuinely captures the alias topic — e.g., 'mentor through code review' — keep narrowly.)

## Action plan

Suggested order, cheapest first.

**1. Apply hard drops** (zero-hit keywords). ~830 keywords. Risk: near-zero — if a keyword fires on 0 of 19,433 SWE postings, it cannot affect headline rates. Saves regex overhead and reduces noise in the spec.

**2. Apply cross-list reconciliation.** Use the 6 rules above to deduplicate. Update each topic's `notes` field to point to the canonical home of any aliased phrase.

**3. Apply guards** to flagged false-positive keywords. The largest contributors by topic are mentorship (90), process_scaffolding (40), context_infrastructure (32), legacy_stack (29). For each guard, decide: drop, narrow to specific phrase, or wrap with negative-lookahead at count time.

**4. Apply additions.** 86 candidates, all grounded in corpus snippets. Cheap to apply.

**5. Re-run calibration.** `./.venv/bin/python paper/vocab_lists/calibration/run_calibration.py`. Expected outcome: dramatically smaller per-topic JSONs (zero-hit entries gone), saturation outliers wrapped, no in-list duplicates.

**6. Layer-4 human grounding** (out of agent scope). Hand-label 100–200 postings per topic on a binary 'does this posting express ⟨topic⟩' rubric and compute concept-level F1 against the keyword-density labels. This is the alt-test the paper's appendix already commits to.

## Files

- `vocab_lists.json` — original consolidated vocab (input).
- `calibration/<slug>_calibration.json` — per-topic per-keyword corpus hits and examples.
- `calibration/summary.json` — top-level calibration summary.
- `calibration/collisions.json` — full cross-list collision index.
- `calibration/edit_recommendations.json` — machine-readable consolidated edits (this run).
- `calibration/review.md` — this document.
- `calibration/run_calibration.py` — re-runnable calibration script.
- `calibration/consolidate_review.py` — consolidates per-topic reviews into review.md + edit_recommendations.json.
