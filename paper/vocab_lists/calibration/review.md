# Vocab Lists — Calibration Review

_Generated 2026-05-06T01:05:44+00:00._


Calibration sample: **19,433 SWE postings** stratified across `kaggle_arshkon/2024-04`, `kaggle_asaniczka/2024-01`, `scraped/2026-04`, matched against `description_core_llm`.


## Executive summary

- Input vocabulary: **2,424 keywords** across **88 core concepts** in **8 topics**.
- Recommended **drops:** 168 keywords (~7% of input — mostly zero-hit entries).
- Recommended **guards:** 149 keywords with high false-positive risk visible in example matches.
- Recommended **additions:** 109 new keywords surfaced from corpus inspection.
- **Cross-list collisions:** 168 keywords appear in 2+ topics; 6 reconciliation rules proposed below.

### Per-topic state

| Topic | Concepts | Keywords | Zero-hit | Drop | Guard | Add | Concept hit-rate range |
|---|---:|---:|---:|---:|---:|---:|---|
| **people_management** | 10 | 147 | 2 | 17 | 26 | 9 | 0.0% — 23.9% |
| **orchestration** | 10 | 208 | 0 | 9 | 11 | 11 | 0.0% — 54.9% |
| **verification** | 9 | 352 | 4 | 8 | 14 | 17 | 5.5% — 40.8% |
| **mentorship** | 11 | 252 | 0 | 30 | 34 | 17 | 0.8% — 22.4% |
| **performance** | 14 | 512 | 115 | 52 | 18 | 14 | 0.7% — 39.0% |
| **process_scaffolding** | 11 | 350 | 2 | 12 | 18 | 11 | 9.2% — 53.1% |
| **legacy_stack** | 12 | 220 | 2 | 17 | 14 | 18 | 0.6% — 3.6% |
| **context_infrastructure** | 11 | 383 | 3 | 23 | 14 | 12 | 2.4% — 34.2% |

## Per-topic findings

### People-management markers (`people_management`)

This vocabulary list is NOT trustworthy as currently constructed: roughly half of all flagged postings will be false positives driven by four polluting tokens. 'supervised' (n=131) is dominantly the ML term 'supervised learning'; 'supervision' (n=648) is overwhelmingly the autonomy phrase 'minimal/limited supervision' (i.e., the OPPOSITE meaning); 'oversee/oversight/oversees' (n=1,048 combined) is mostly project/process oversight rather than people; and 'team of' (n=1,527) catches 'collaborate with a team of' / 'join a team of' boilerplate. The 'Coaching, mentoring, and career development' concept (24% hit rate) is the largest concept by volume but conflates informal mentorship and learning-and-development benefits with the people-management responsibilities the topic definition explicitly tries to isolate. The high-precision signals (direct reports, engineering manager, people management/leader, weekly 1:1, performance evaluations/appraisals/reviews, compensation recommendations, manage/managing a team of, hiring decisions, headcount, Tech-Lead-Manager) are present but drowned out and need to become the load-bearing terms for the measurement. After the fixes below the list should be usable; without them, every reported people-mgmt rate will be inflated by an order of magnitude.

_17 drops · 26 guards · 9 adds · 5 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Coaching, mentoring, and career development` — 23.9%
- `Supervisory verbs` — 10.5%
- `Team leadership` — 8.8%
- `Team-size signals` — 7.9%
- `People management role` — 2.3%
- `Performance management` — 0.7%
- `Headcount and hiring authority` — 0.5%
- `Direct reports` — 0.4%
- `1:1 meetings` — 0.2%
- `Termination, hiring/firing authority` — 0.0%

**Top guards** (false-positive risks worth fixing):
- `supervisory` (Supervisory verbs): 'no supervisory responsibilities' / 'does not have supervisory responsibilities' (negation) appears in JD boilerplate (e.g., Application Support Specialist, Solutions Architect/Project Manager). → _guard_: Add a negation guard: drop matches within 30 chars of 'no '/'does not have '/'without' preceding the term.
- `supervise` (Supervisory verbs): 'limited supervision and may supervise/lead others' — conditional/optional framing. Also 'supervise others' is fine but rare without context. → _guard_: Require co-occurrence with a people noun ('staff'|'team members'|'engineers'|'developers'|'employees'|'reports') within 8 tokens; drop bare 'supervise'.
- `supervises` (Supervisory verbs): 'supervises the design, development, and maintenance' / 'supervises change control processes' — process not people. → _guard_: Require people object: 'supervises ' + ('staff'|'engineers'|'team'|'employees'|'developers'|'analysts'|'reports').
- `team leader` (Team leadership): 'project team leader' as a project-management role; 'may assume the role of team leader as assigned' is rotational/informal — and it often refers to OTHER people in the org. → _guard_: Require first/second-person framing ('as a team leader you', 'serve as team leader of') OR people-management cooccurrence; drop bare third-party mentions.
- `team leads` (Team leadership): Plural form mostly refers to OTHER people the candidate collaborates with: 'collaborate with team leads', 'partner with product owners, tech leads, designers' — not the role itself. → _guard_: Drop, or require possessive/role framing ('as team leads', 'team leads will'); plural form is dominated by collaborator references.

**Top suggested additions** (grounded in corpus snippets):
- `people-management experience` → People management role
- `engineering manager role` → People management role
- `performance discussions` → Performance management
- `promotion assessments` → Performance management
- `promotion decisions` → Performance management

**Concept-level recommendations:**
- `Coaching, mentoring, and career development`: This concept is the highest-volume in the topic (24% hit rate, n=4,640) but its top keywords (mentor 13%, mentoring 7.5%, mentorship 3.8%, coach 2.1%) are explicitly excluded by the topic definition as informal mentorship / senior-IC behavior. As constructed, this concept measures senior-IC duties, not people management. → _Either (a) drop the entire concept from people-management and move it under a separate 'senior-IC mentorship' topic, or (b) narrow to manager-only phrasings: 'career development of [their|your team]', 'develop talent', 'talent development', 'grow engineers' (with possessive), 'develop your team', 'grow your team', 'individual development plan'. Drop bare mentor*/coach* tokens entirely._
- `Supervisory verbs`: Three of the highest-volume keywords are net-negative for the topic: 'supervised' is dominantly 'supervised learning' (ML term), 'supervision' is dominantly 'minimal supervision' (autonomy framing — opposite meaning), and 'oversee/oversight' is dominantly process oversight. The concept as a whole has a 10.5% hit rate but most of those are false positives. → _Split into (a) 'Direct supervisory verbs' = supervise/manage with required people-object cooccurrence and (b) drop oversee/oversight/supervision/supervised entirely from people-management (these are project-leadership / ML / autonomy phrases). After the prune, expected hit rate drops to ~1-2% but precision rises sharply._
- `Team leadership`: Concept conflates IC-senior titles ('lead software engineer', 'technical lead', 'lead engineer', 'lead developer') with actual team-leadership phrasings. The IC titles dominate by volume (lead software engineer n=294, technical lead n=338, tech lead n=181) and most are pure technical-leadership roles without people-management duties. The topic definition acknowledges 'tech-lead/lead-engineer hybrid roles that imply some people responsibility' but the keyword list does not enforce the 'imply some people responsibility' part. → _Split into (a) 'Title-only team-lead markers' (team lead, team leader, head of engineering) treated as WEAK signals requiring confirmation from another concept hit, and (b) 'Action-verb team leadership' (lead a team of, leading a team of, manage a team of, managing a team of, led a team of) as STRONG signals. Move all 'lead software engineer'/'lead developer'/'lead engineer'/'technical lead'/'tech lead'/'engineering lead' OUT of people-management and into a separate 'technical leadership' topic._
- `Team-size signals`: 'team of' alone is far too generic; it captures collaborator/peer-team references, not management scope. Only the verb-anchored variants ('manage/managing/lead/leading a team of') signal management. → _Drop bare 'team of'. Keep only verb-anchored variants and explicit numeric/size phrasings (team size, span of control, headcount of, team of [N]). Fix the broken regex 'team of \d+' (currently zero hits) — replace with proper regex pattern._
- `1:1 meetings`: Bare '1:1' catches non-meeting uses ('1:1 financial planners', '1:1 basis', '1:1 feature implementations'); concept hit rate is already low (0.2%) so it does not poison the topic, but it adds noise. → _Replace bare '1:1' with phrase forms only: '1:1 meeting(s)', '1:1s with', 'weekly 1:1', 'regular 1:1'. Keep one-on-one(s) as-is._

### Orchestration (`orchestration`)

The list is structurally strong but contaminated by several high-volume tokens that fire on entirely unrelated meanings: 'MCP' is dominantly a Marketing Cloud / Microsoft Certified Professional / Mariner Credentialing acronym in 2024 data, 'HITL' is overwhelmingly Hardware-in-the-Loop (not Human-in-the-Loop), and 'A2A' frequently means B2B-style Application-to-Application integration. The Agent Memory concept is severely polluted: 'memory management', 'memory system', 'memory systems', and 'short-term memory' all match C/C++ memory management and DDR/LSTM hardware/ML contexts and have nothing to do with agent state. The 'architect' / 'architects' tokens fire mostly on job titles (Java Architect, ServiceNow Architect) rather than the activity of architecting, inflating Software Architecture's hit rate; the inflected verb forms ('architecting', 'architected') are clean. Genuine emerging-practice signals (agent harnesses, agentic primitives, sub-agents, supervisor-worker, spec-driven, CLAUDE.md) are all real but extremely rare, which is a substantive finding rather than a list defect; the 2026-04 stratum is doing the work for those. Several useful agent-era keywords are missing from snippets: 'Bedrock AgentCore', 'Strands Agents', 'AgentCore', 'Foundry agents', 'Strands', 'planner/executor', 'tool dispatch', 'agentic loops', 'Agent Core', 'Glean', 'ACP', 'Codex'.

_9 drops · 11 guards · 11 adds · 4 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Software Architecture and System Design` — 54.9%
- `Workflow Design and Pipeline Orchestration` — 39.9%
- `Specifications and Requirements Authoring` — 17.3%
- `Task Decomposition and Planning` — 12.7%
- `Agent Harnesses and Coding Agents` — 5.7%
- `Context Engineering and Prompt/Spec Authorship for Agents` — 2.6%
- `Multi-Agent Coordination` — 1.2%
- `Agent Memory and State Management` — 1.0%
- `AI Tool and Agent Evaluation/Steering` — 0.9%
- `Repository Instructions and Agent Configuration Files` — 0.0%

**Top guards** (false-positive risks worth fixing):
- `architect` (Software Architecture and System Design): Dominantly fires on job titles: 'Java architect', 'AWS Cloud Architect', 'ServiceNow architect', 'Solutions Architect', 'Enterprise Architect', 'Data Architect', etc. The 2,644 hits massively overstate architecture-as-activity. → _guard_: Require verb-form context (e.g., 'architect and build', 'architect, design, and') OR exclude when token sits inside a job-title-style phrase ('Cloud Architect', 'Solutions Architect', '* Architect *' role line). Prefer the inflected forms 'architecting' / 'architected' which are unambiguous.
- `architects` (Software Architecture and System Design): Plural overwhelmingly refers to people in titles: 'collaborate with architects', 'principal software engineers and architects', 'Enterprise architects and developers'. Almost never the verb. → _guard_: Drop or require co-occurrence with a verb subject ('architects the system', 'architects solutions'). Otherwise a people-noun, not architecture activity.
- `architecture` (Software Architecture and System Design): Mostly genuine but a sizable share is 'microservices architecture'/'containerization architecture' as a stack noun rather than design activity. Also 'orchestration architecture' overlaps with the Workflow concept. → _guard_: Accept as-is; the construct is broad by design. Optionally require co-occurrence with a design verb ('design the architecture', 'architecture decisions', 'architectural patterns') if a stricter signal is needed.
- `A2A` (Multi-Agent Coordination): One in five inspected hits is 'A2A/B2B Integration' (legacy enterprise application-to-application integration), not Agent-to-Agent protocol. The remaining four are genuine. → _guard_: Require co-occurrence with at least one of: 'agent', 'protocol', 'MCP', 'LangChain', 'LangGraph', 'AutoGen', 'multi-agent'. Drop matches that pair with 'B2B' or 'EDI' or 'integration scenarios'.
- `breakdown` (Task Decomposition and Planning): Includes 'Tech Breakdown | 40% Azure 40% Application Support' role-allocation tables and 'Technical Breakdown | 80% Backend' job-listing breakdowns, which are not task decomposition. → _guard_: Require co-occurrence with 'task', 'work', 'project', 'requirements', 'problem', or 'feature'. Drop matches in tabular role-percentage contexts ('Tech Breakdown |' followed by percentages).

**Top suggested additions** (grounded in corpus snippets):
- `MCP server` → Agent Harnesses and Coding Agents
- `MCP servers` → Agent Harnesses and Coding Agents
- `AgentCore` → Agent Harnesses and Coding Agents
- `Bedrock AgentCore` → Agent Harnesses and Coding Agents
- `Strands Agents` → Agent Harnesses and Coding Agents

**Concept-level recommendations:**
- `Agent Memory and State Management`: Concept is overwhelmed by classical-engineering false positives: 'memory management' (C++/Node), 'memory system'/'memory systems' (DDR validation), 'short-term memory'/'short term memory' (LSTM). Removing those four leaves only 6 keywords with very low individual volume, but the resulting hit rate would actually reflect agent memory rather than be polluted by an order of magnitude of unrelated DDR/LSTM/C++ noise. → _Drop the four polluted general-memory keywords. Replace with agent-specific phrasing: 'memory store', 'episodic memory', 'memory layer', 'persistent agent memory', 'session memory', 'conversation memory', 'checkpointing' (LangGraph). Keep 'agent memory', 'agentic memory', 'long-term memory' (with guard), 'context retention', 'stateful agents', 'agent state'._
- `Repository Instructions and Agent Configuration Files`: Only 4 keywords, hit rate 0.0004, and 'agent configuration' is contaminated by ServiceNow Virtual Agent. Concept will struggle to cross the visibility threshold for any quantitative analysis. → _Add: 'rules file', 'rules files', 'AGENTS.md', '.claude' (directory), 'system prompt file', 'agent rules', 'project rules', 'context file'. Tighten 'agent configuration' with the guard above. Concept is meaningful but data-limited - acknowledge in paper that this is an emerging signal that 2026 corpus barely captures._
- `AI Tool and Agent Evaluation/Steering`: 'HITL' is unrecoverable (Hardware-in-the-Loop dominance) and should be dropped, leaving the human-supervision facet under-keyworded. The concept also conflates LLM-output evaluation (LLM evals, evals, AI evaluation) with human-supervision-of-agents (review AI-generated code, human-in-the-loop, AI guardrails). → _After dropping 'HITL', add steering-specific keywords observed in snippets: 'agent observability', 'LangSmith' (cross-list aware), 'agent tracing', 'agent telemetry', 'guardrails' (separate from 'AI guardrails'), 'safety controls', 'agent feedback loop'. Consider splitting into two sub-concepts if granularity matters: (a) Output evaluation/evals frameworks, (b) Human steering/review/guardrails._
- `Software Architecture and System Design`: The 'architect' / 'architects' tokens are job-title-dominated and inflate this concept's hit rate (0.5486) well above the actual architectural-design activity. The concept appears to dominate the topic but a substantial fraction of that signal is noise from job-title self-references. → _Either drop 'architect' singular and 'architects' plural and rely on 'architecting'/'architected'/'architecture'/'architectural', or apply the guard described above. Recompute concept_hit_rate after the fix - the true architectural-activity signal is likely closer to 0.40 than 0.55._

### Verification (`verification`)

The list is largely well-targeted but has a small number of catastrophic false positives that should be removed outright: 'TypeScript' (a programming language, not a static-analysis tool, 1847 hits), 'Phoenix' (overwhelmingly the city of Phoenix, AZ, 62 hits with only ~5 referring to Arize Phoenix), and 'HITL' (almost entirely 'Hardware-in-the-loop' avionics/embedded testing rather than human-in-the-loop AI). The QA/validation concept has heavy contamination from 'validation/validate/validating' against statistical model validation, requirement validation, and data validation contexts that do not signal QA-style verification expectations and need explicit guards or scoping. The AI-eval concept conflates classical performance benchmarking with LLM evaluation: 'benchmark', 'benchmarks', 'benchmarking', and 'model evaluation' fire mostly on perf benchmarking and classical ML model evaluation, not on AI output verification, dragging the concept's apparent prevalence upward. Coverage gaps: common test frameworks (JUnit, Selenium, Cypress, Jest, Playwright, Cucumber, TestRail) and concrete LLM-eval frameworks (RAGAS, TruLens, DeepEval) repeatedly appear in snippets but are absent from the list.

_8 drops · 14 guards · 17 adds · 5 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `CI/CD and quality gates` — 40.8%
- `QA and validation` — 29.0%
- `Post-deployment observability` — 26.4%
- `Automated testing` — 22.8%
- `Compliance and security verification` — 16.8%
- `Code review` — 16.0%
- `Static analysis and code quality tooling` — 15.2%
- `Reproducibility and artifact proof` — 10.3%
- `Evaluations and AI output verification` — 5.5%

**Top guards** (false-positive risks worth fixing):
- `validation` (QA and validation): Heavily fires on 'data validation', 'requirement validation', 'model validation' (statistical), 'design validation', 'validation of customer needs' — i.e., systems-engineering/data-quality validation rather than software QA. Of the top 5 snippets, 4 are non-QA contexts. → _guard_: Require co-occurrence with software-test context: keep only when the surrounding window contains one of {test, QA, software, code, build, deploy, release, defect}; otherwise treat as out-of-topic.
- `validate` (QA and validation): Captures 'validate assumptions', 'validate hypotheses', 'validate model effectiveness', 'validate results with customer' — generic verb usage outside QA/testing. → _guard_: Same window requirement as 'validation'; or restrict to phrases like 'validate test', 'validate functionality', 'validate fixes', 'validate code'.
- `validating` (QA and validation): Fires on 'validating data sources', 'validating statistical models', 'validating results' — research/analytics context, not software QA. → _guard_: Require nearby test/QA/defect/code anchor.
- `verification` (QA and validation): Co-occurs heavily with systems-engineering 'verification and validation' (V&V) for hardware/defense/medical-device requirements rather than software test verification. Examples include 'design verification', 'data verification testing', 'systems integration, verification and validation, cost and risk'. → _guard_: If matched alongside 'systems engineering', 'requirements traceability', or 'medical device', consider it systems-engineering V&V rather than software QA. Either tag separately or require software/code/test anchor in the window.
- `QA` (QA and validation): Often refers to the QA function/role (cross-functional team mention) rather than QA activity, e.g. 'product management, design, and QA' — informative but not necessarily an expectation that this engineer does QA. → _guard_: Acceptable as is for topic prevalence, but in role-level analysis differentiate 'QA function/team' mentions from 'do QA / write tests / QA process' phrasings.

**Top suggested additions** (grounded in corpus snippets):
- `Selenium` → Automated testing
- `JUnit` → Automated testing
- `Cypress` → Automated testing
- `Jest` → Automated testing
- `Playwright` → Automated testing

**Concept-level recommendations:**
- `Static analysis and code quality tooling`: Including 'TypeScript' as static analysis is a category error: it is a language, not a checker. The concept's hit rate is inflated by ~30% because of this single keyword (1847 of 2950 concept-postings include 'TypeScript'). → _Drop 'TypeScript'. If type-checking-as-verification is in scope, replace with explicit type-checking activity phrases: 'strict types', 'type safety', 'type checks', 'static types' (in addition to existing 'type checking' and 'mypy'). Recompute the concept hit rate after the drop._
- `Evaluations and AI output verification`: The concept currently mixes (a) classical-ML evaluation vocabulary (benchmark/benchmarking/model evaluation) and (b) LLM/GenAI-specific evaluation (RAGAS, LangSmith, hallucination, guardrails, LLM-as-a-judge). These are distinct phenomena in the labor-market signal: (a) is decade-old DS/ML, (b) is post-2023 GenAI. → _Either split into two sub-concepts ('Classical ML evaluation' vs 'LLM/AI output verification'), or scope the concept tightly to LLM/AI-era and require an LLM/AI/agent/prompt/GenAI anchor for ambiguous keywords (benchmark*, model evaluation*, evaluations, eval). This will give a much more accurate post-2023 GenAI signal._
- `QA and validation`: 'validation/validate/validating/verification' are heavily polluted by systems-engineering V&V, statistical model validation, and requirements validation — domains that share the vocabulary but represent different work. This inflates the concept's hit rate disproportionately for medical-device, defense, and data-engineering postings. → _Either (i) require a software-test anchor (test/QA/code/defect/build) within a small window for these four keywords, or (ii) split out 'systems-engineering V&V' as a separate concept so the QA concept reflects software-QA expectation specifically._
- `Post-deployment observability`: Bare verbs 'monitor'/'monitored'/'monitors' and the noun 'metrics' are too generic and pull in security-monitoring, business-metrics, and unrelated 'monitor server' usage. The concept's 26% hit rate is partly an artifact of these tokens. → _Drop bare 'monitor'/'monitors'/'monitored' or restrict to compound phrases ('production monitoring', 'service monitoring', 'health monitor'). For 'metrics', require co-occurrence with operational anchors ({alert, dashboard, SLI, SLO, latency, error rate, OpenTelemetry, Prometheus, logging})._
- `Reproducibility and artifact proof`: The concept conflates two different activities: (a) producing artifacts as evidence of correctness (logs/screenshots/traces submitted with PRs) and (b) generic logging/tracing infrastructure (which overlaps with observability). 'logging', 'logs', 'log', 'tracing', 'traces' are heavily used in the latter sense and largely double-count with the observability concept. → _Either narrow this concept to true artifact-as-evidence vocabulary ('reproduce', 'reproducibility', 'reproducible', 'audit trail', 'audit log', 'screenshots', 'recording', 'evidence'), moving 'logging/logs/tracing/traces' wholly to observability; or accept the overlap and document that this concept is largely co-extensive with observability infra._

### Mentorship markers (`mentorship`)

Direct mentorship vocabulary (mentor*, mentoring, mentorship) is solid but should be paired with a teaching-direction guard since 'mentor'/'coach' frequently target customers, end-users, partners, or non-engineers. The 'Guidance and influence' concept is severely overbroad: bare tokens like 'guidance' (2,097), 'technical guidance' (518), 'consulting' (493), 'advisor' (219), 'advocate' (318), and 'technical leadership' (1,048) overwhelmingly fire on customer/stakeholder advisory or generic seniority signals rather than peer teaching, inflating the concept hit rate to 22%. The 'Debugging and code-reading as a craft' concept is misclassified: 'debug'/'debugs'/'debugging' fire on baseline coding duties, not on debugging-as-mentorship; only the 'walkthroughs' family genuinely fits. Several concepts (Onboarding, Code review as teaching, Architecture and design guidance, Teaching) contain keywords that capture generic process language ('onboarding', 'review code', 'training materials', 'establish best practices', 'technical direction') with little or no mentorship framing. Recommend tightening the high-volume buckets via co-occurrence guards (require nearby mentor/junior/team-member/teach/grow language) and dropping or scoping ~30 keywords whose snippets show systematic false positives.

_30 drops · 34 guards · 17 adds · 8 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Guidance and influence` — 22.4%
- `Direct mentorship language` — 21.8%
- `Debugging and code-reading as a craft` — 17.6%
- `Explicit learning culture` — 6.4%
- `Teaching and knowledge transfer` — 6.1%
- `Code review as teaching` — 5.1%
- `Coaching and developing engineers` — 4.6%
- `Architecture and design guidance` — 4.5%
- `Onboarding and ramp-up` — 2.8%
- `Pair / mob / ensemble programming` — 1.5%
- `Communication and feedback skills` — 0.8%

**Top guards** (false-positive risks worth fixing):
- `mentor` (Direct mentorship language): Occasionally describes mentoring 'team members on ServiceNow', 'non-technical end users', or appears as the noun 'mentors' in pure noun phrases without engineer object. → _guard_: Require object/context to be engineering-relevant: nearby tokens engineer*|developer*|junior*|peer*|team member*|other engineer* within ~6 words; OR the phrase 'mentor and X' where X in {coach, guide, develop, grow, train, support} also OK.
- `mentors` (Direct mentorship language): As noun ('passionate mentors') without verb behavior; ambiguous when standalone. → _guard_: Accept only if used as a verb (preceded by subject pronoun/'who'/'and') or followed by an engineer object; reject pure noun-phrase listings.
- `coach` (Coaching and developing engineers): FPs: 'coach team members on ServiceNow development', 'coach business stakeholders', 'coach non-technical end users', 'player/coach' sports metaphor. → _guard_: Require object to be engineer*|developer*|junior*|team*|peer*|other(s)* within window; reject when followed by 'business stakeholders', 'end users', 'non-technical', 'customers', 'students'.
- `coaching` (Coaching and developing engineers): FPs: 'receive coaching from senior personnel' (being coached), and 'provide coaching, training, and mentoring to Software Quality Engineers' (OK). → _guard_: Require role to be giver, not receiver: nearby 'provide|deliver|give|offer' coaching; reject 'receive coaching', 'under coaching of'.
- `coaches` (Coaching and developing engineers): Possessive 'Coaches' technology communities' refers to industry-of-practice coaching at companies like Discover; ambiguous. → _guard_: Require an engineer/team-of-engineers object: 'coaches the team', 'coaches engineers', 'coaches developers'.

**Top suggested additions** (grounded in corpus snippets):
- `make others better` → Communication and feedback skills
- `make engineers better` → Communication and feedback skills
- `ping-pong programming` → Pair / mob / ensemble programming
- `paired / mobbing` → Pair / mob / ensemble programming
- `pairing and mobbing` → Pair / mob / ensemble programming

**Concept-level recommendations:**
- `Debugging and code-reading as a craft`: The bare 'debug*' family (debug/debugs/debugged/debugging) accounts for nearly all 3,414 hits and almost never carries mentorship/learning framing — it's a baseline coding-duty verb. The concept's claimed framing (debugging as a teaching activity) is not detectable from these tokens alone. → _Either drop the concept entirely, or restrict it to bigrams that carry the teaching/learning sense: keep only 'code walkthroughs', 'walkthroughs', 'reading code', 'read code', 'debugging together', 'pair-debugging', 'walk through the code', 'live debugging session'. Drop bare debug/debugs/debugged/debugging. Without this fix the concept will dominate Mentorship hit-rate purely through generic JD verbiage._
- `Guidance and influence`: Concept is presently a catch-all that conflates (a) senior IC mentoring engineers via influence with (b) customer/stakeholder advisory roles ('trusted advisor', 'consulting engineer', 'technical advisor for cloud solution') and (c) generic seniority titles ('technical leadership', 'engineering leadership', 'thought leader'). The concept hit rate (22.4%) is artificially inflated by signals unrelated to engineer development. → _Split into two sub-concepts: (1) 'Influence-without-authority over engineers' — keep 'lead by example', 'leading by example', 'lead through influence', 'influence without authority', 'role model', 'be a role model', 'set an example', 'champion engineering', 'support junior engineers', 'guide junior', 'guide engineers', 'guide the team', 'guide others'; (2) move 'consulting', 'advisor*', 'trusted advisor', 'technical advisor', 'thought leader*', 'advocate' OUT (they belong to customer-advisory or seniority topics, not mentorship). Apply human-object guard to remaining 'guidance' phrases._
- `Code review as teaching`: Most keywords in this concept ('review code', 'reviewing code', 'peer review*', 'architectural review*', 'design discussions', 'provide feedback') overwhelmingly capture review-as-correctness-gate rather than review-as-teaching, duplicating Verification signal. Definition explicitly notes the overlap but the keyword list does not enforce the disambiguation. → _Treat the concept as 'review + teaching framing in same sentence'. Either (a) require a teaching/feedback/mentor/grow/learning co-occurrence as a hard rule on review-family tokens, or (b) keep only the explicitly teaching-marked phrases — 'thoughtful code review', 'review that teaches', 'constructive feedback' (with giver-guard), 'actionable feedback', 'thoughtful feedback', 'code review and mentorship' bigrams. Remove or guard the bare review-family tokens._
- `Onboarding and ramp-up`: Bare 'onboard'/'onboarding' tokens are heavily polluted by product/data/integration onboarding ('onboarding new datasets', 'onboarding flow', 'onboard integrations', 'clients onboard', 'employee lifecycle onboarding product'). The 'first 30/90 days' tokens hit certification deadlines. → _Require human object on all onboarding tokens: 'onboarding (new hires|engineers|developers|colleagues|team members|new team members|new employees)'. Require ramp/learn/plan/orientation context on '30/90 days' tokens; reject when followed by '...of employment to obtain certification'._
- `Coaching and developing engineers`: Top-volume 'coach'/'coaching'/'coaches' tokens fire on 'coach business stakeholders', 'coach end users', 'player/coach' metaphor, and on 'receive coaching' (the engineer is the recipient, not the giver). 'Raise the bar' is almost always quality-oriented, not engineer-oriented. → _Apply (a) engineer-object guard to coach* tokens (engineer*|developer*|junior*|peer*|team member*|other(s)*); (b) giver-role guard (reject 'receive', 'under coaching of'); (c) require 'team' or engineer-context proximity for 'raise the bar' family. Drop 'support the growth' bare token in favor of 'support the growth of (engineers|peers|team|others|colleagues|teammates)'._

### Performance & deep technical understanding (`performance`)

The 'performance' topic shows severe semantic drift in three concepts and good signal in the technical core. depth_claim_language is the worst offender: terms like 'fundamentals', 'fluent in', 'fluency in', 'mastery', 'expert in', 'expert level', 'subject matter expert', 'SME', 'from scratch', and 'ground up' fire on commodity job-posting boilerplate (e.g., 'mastery of JavaScript', 'fluent in Git workflows', 'building data pipelines from scratch', 'SME for upper management', 'compliance SME'), inflating the topic hit-rate to 39 percent without indicating depth-of-understanding demand. low_level_systems_programming over-includes embedded/firmware roles ('systems engineer', 'systems engineering', 'firmware', 'embedded systems', 'embedded software') which is its own labor-market category and crowds out true low-level perf work; 'assembly' and 'SSE' have catastrophic literal-match collisions (camera assembly, Systems Security Engineering). scaling_and_efficiency includes runaway commodity terms ('efficiency', 'scalability', 'highly scalable', 'efficient code') that fire on generic JD copy. By contrast, the technical concepts (latency_throughput_scale numeric framings, profiling tools, compiler/kernel internals, GPU/FPGA, lock-free, query optimization, Iceberg/compaction) work cleanly and are the topic's actual signal — the depth-claim concept should be narrowed to depth-framings only, and a 'performance-critical work / from-first-principles design' subset should be preserved.

_52 drops · 18 guards · 14 adds · 10 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `depth_claim_language` — 39.0%
- `scaling_and_efficiency` — 23.2%
- `latency_throughput_scale` — 14.6%
- `performance_optimization_general` — 13.9%
- `low_level_systems_programming` — 12.7%
- `distributed_systems_internals` — 8.7%
- `profiling_and_benchmarking` — 6.7%
- `algorithmic_optimization` — 5.4%
- `concurrency_parallelism` — 5.1%
- `hardware_aware_programming` — 2.9%
- `compilers_and_runtimes` — 1.6%
- `os_kernel_internals` — 1.6%
- `database_storage_internals` — 1.3%
- `networking_internals` — 0.7%

**Top guards** (false-positive risks worth fixing):
- `performance engineering` (performance_optimization_general): Phrase 'enterprise management and performance engineering concepts' is a recurring DevSecOps/SRE boilerplate from one or two JD templates that contributes most hits. → _guard_: Require co-occurrence with profiling/benchmarking/optimization terms within +/- 15 tokens, or drop the bigram and rely on more specific phrases.
- `tracing` (profiling_and_benchmarking): Hits include 'requirements tracing' (regulatory), 'data tracing' (lineage), and project-level tracing — not distributed tracing. → _guard_: Require neighboring 'distributed tracing', 'OpenTelemetry', 'Jaeger', 'Zipkin', or 'instrumented' OR collapse to 'distributed tracing' as a bigram.
- `instrumentation` (profiling_and_benchmarking): Many hits are physical 'PLC instrumentation', 'field instrumentation' (industrial automation), or display-instrumentation hardware — not software instrumentation. → _guard_: Require co-occurrence with 'metrics' / 'tracing' / 'observability' / 'profil*' / 'OpenTelemetry' within the same posting.
- `benchmark` (profiling_and_benchmarking): 'Setting a benchmark for engineering excellence', 'benchmark different ML models' — figurative or evaluation-style. → _guard_: Require neighboring 'performance', 'latency', 'throughput', 'profile', 'load' OR drop singular and keep 'benchmarks'/'benchmarking' which are stronger.
- `real time` (latency_throughput_scale): 314 hits include 'real time data', 'real time insights' (BI/analytics framing), 'real-time Human Capital insights' (AI marketing) — not perf/latency commitment. → _guard_: Drop space-form OR require neighboring 'latency', 'embedded', 'streaming', 'low-latency'; the hyphenated form is more reliable but also needs guarding.

**Top suggested additions** (grounded in corpus snippets):
- `perf` → profiling_and_benchmarking
- `Google Benchmark` → profiling_and_benchmarking
- `JMH` → profiling_and_benchmarking
- `bcc tools` → profiling_and_benchmarking
- `performance-oriented` → performance_optimization_general

**Concept-level recommendations:**
- `depth_claim_language`: The concept currently mixes three things: (a) genuine depth-of-understanding framings ('deep understanding', 'deep expertise', 'in-depth knowledge', 'first principles', 'rigorous understanding'), (b) generic skill-proficiency boilerplate ('mastery of X', 'fluent in X', 'expert in X', 'fundamentals of X'), and (c) greenfield-project framings ('from scratch', 'ground up'). All three fire on virtually every JD, inflating the topic to ~39% hit-rate without measuring the construct. → _Restrict to category (a) only: keep 'deep understanding/knowledge/expertise/familiarity', 'in-depth understanding/knowledge', 'thorough understanding/knowledge', 'rigorous understanding', 'first principles', 'from first principles', 'first-principles', 'intimately familiar', 'intimate knowledge', 'under the hood', 'deeply understand'. Drop all (b) and (c) keywords. Consider also dropping unweighted 'solid understanding' / 'strong understanding' which are universal JD-opener phrasing — they signal nothing distinctive about depth-demand._
- `low_level_systems_programming`: The concept conflates two labor markets: (1) low-level performance/systems work (assembly, SIMD, memory model, lock-free) which is the topic's intent, and (2) embedded/firmware engineering (medical devices, avionics, automotive) which is its own category and brings in 1500+ postings dominating the concept. → _Either split into two concepts ('low_level_systems_programming' for SIMD/atomics/memory-model/assembly + 'embedded_firmware' separate), or drop 'firmware', 'embedded systems', 'embedded software', 'systems engineer', 'systems engineering' from this concept and rely on the precise low-level keywords. Also remove the literal-collision keywords 'assembly', 'SSE'._
- `scaling_and_efficiency`: Universal JD verbiage ('scalability', 'efficiency', 'highly scalable', 'efficient code') accounts for the bulk of the 23% concept hit rate but carries no perf-engineering signal. → _Drop 'scalability', 'efficiency', 'highly scalable', 'efficient code'. Keep numeric/quantified forms ('scale to millions', 'scale to billions', 'massive scale', 'horizontal scaling', 'vertical scaling') and resource-specific terms ('CPU utilization', 'memory footprint', 'memory pressure', 'memory leaks', 'capacity planning'). 'Cost optimization' / 'cost efficiency' are FinOps-coded and should arguably be moved to a finops/cost concept rather than performance._
- `performance_optimization_general`: 'high performance' / 'high-performance' fire on culture-deck and team-naming boilerplate ('high-performance team', 'high-performance, diverse thought') as much as on actual high-performance computing. → _Keep precise variants 'high-performance computing' and 'HPC' only; drop bare 'high performance' / 'high-performance' OR guard them by requiring co-occurrence with 'computing', 'system', 'application', 'database', 'code' within +/- 5 tokens (excluding 'team', 'culture', 'environment')._
- `distributed_systems_internals`: Generic terms ('consensus', 'replication', 'partitioning') fire heavily on non-internals usage (stakeholder consensus, ETL replication, table partitioning). Meanwhile half the keywords are zero-hit theoretical-CS terms (linearizability, CAP theorem, vector clocks, CRDT singular form, MVCC, LSM tree). → _For 'consensus' and 'replication', require neighboring algorithmic context ('Raft', 'Paxos', 'log', 'leader', 'quorum') OR drop them. Drop or merge zero-hit theoretical terms: keep the few that do fire ('CRDTs', 'consistency models', 'two-phase commit', 'WAL') and consider that this concept is a low-recall by-design signal._

### Process-scaffolding markers (`process_scaffolding`)

The process_scaffolding list is structurally sound for the head terms (agile/scrum/sprint/requirements/specifications/SDLC/governance/coordination/scheduling/Jira), but it is contaminated by a cluster of short ambiguous tokens that match unrelated tech (SAFe -> 'safe', LeSS -> 'less' / LESS CSS preprocessor, SM -> AWS Secrets Manager, PO -> SAP/Oracle Purchase Order, NFR -> the company NFR, ART -> 'state of the art', PLC -> Programmable Logic Controller, XP -> Windows XP, TPM -> Trusted Platform Module, Linear -> 'linear algebra', PM -> '5 PM' / Portfolio Manager, CSM -> ServiceNow Customer Service Management, CAB -> 'Cab Engineering', CPM -> Cost-Per-Mille). A second cluster of V&V terms (validate/validating/validates/validated, verify/verifying/verifies/verified) fires overwhelmingly on data-validation, model-validation, and input-validation snippets rather than process-level V&V. A third cluster of acronym-only forms (DoD, V&V, IV&V, RTM, SRS, ERD, PMI) is largely on-topic but DoD has a heavy second meaning (Department of Defense clearance) that dominates its hits. Finally, the list is missing a small number of mainstream additions visible in snippets (DOORS, PI Planning evidence supports adding 'release train engineer'/'RTE', 'IMS'/'integrated master schedule', 'IMP', 'CMDB', 'Six Sigma', 'agilist', 'grooming').

_12 drops · 18 guards · 11 adds · 5 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Requirements engineering` — 53.1%
- `SDLC and process governance` — 34.2%
- `Agile methodology` — 31.2%
- `Scheduling and milestones` — 23.0%
- `Project / program management roles & tooling` — 19.0%
- `Verification & Validation (V&V)` — 16.4%
- `Project coordination` — 15.8%
- `Scrum framework` — 13.4%
- `Specification authoring` — 11.5%
- `Sprints and iterations` — 9.5%
- `Kanban / Lean / Waterfall and other methodologies` — 9.2%

**Top guards** (false-positive risks worth fixing):
- `SAFe` (Agile methodology): Case-insensitive matching against the common adjective 'safe' ('safe operation', 'safe deployment practices', 'safe and maintainable'). Top examples are all the 'safe' adjective. → _guard_: Match case-sensitively as the all-caps acronym 'SAFe' (with lowercase 'e'), or require a context word: 'SAFe' near 'agile' / 'framework' / 'release train' / 'PI'.
- `SAFe` (Kanban / Lean / Waterfall and other methodologies): Same case-insensitive 'safe' problem as the duplicate in Agile methodology. → _guard_: Same case-sensitive guard or context window. Also consider deduplicating across the two concepts.
- `SAFe agile` (Agile methodology): Case-insensitive matches 'safe Agile' as a phrase ('participate in functional ... in a safe Agile environment'), where 'safe' is the adjective. → _guard_: Require exact case 'SAFe' (capital S, capital A, capital F, lowercase e) or anchor to 'SAFe Agile framework'/'SAFe Agile certification'.
- `LeSS` (Kanban / Lean / Waterfall and other methodologies): 567 hits dominated by the LESS CSS preprocessor ('SASS/LESS', 'Less or Sass') and the comparative 'less experienced'. True Large-Scale Scrum (LeSS) framework hits are essentially zero. → _guard_: Require exact mixed case 'LeSS' (L capital, e lowercase, S capital, S capital) AND a context word like 'scrum'/'framework'/'large-scale'. Without that, the token is unsalvageable.
- `lean` (Kanban / Lean / Waterfall and other methodologies): 'lean DevOps and automation culture', 'lean toward', 'Lean tools include 5S' (manufacturing) - many hits are non-process uses of 'lean'. → _guard_: Require co-occurrence with 'six sigma' / 'startup' / 'principles' / 'manufacturing' / 'kanban' / 'agile' / 'methodology' in the same window, or count only 'Lean' followed by a methodology noun.

**Top suggested additions** (grounded in corpus snippets):
- `DOORS` → Requirements engineering
- `agilist` → Agile methodology
- `SAFe Agilist` → Agile methodology
- `release train engineer` → Project / program management roles & tooling
- `RTE` → Project / program management roles & tooling

**Concept-level recommendations:**
- `Verification & Validation (V&V)`: The bare verb forms (validate, validating, validates, validated, verify, verifying, verifies, verified, validation) are dominated by data-validation, model-validation, and input-validation usage in DS/ML/backend postings, not process-level V&V. As currently defined the concept double-counts a generic engineering activity that is not the systems-engineering V&V of the topic definition. → _Either (a) move the bare verb forms to a guarded list that requires co-occurrence with 'requirements', 'system', 'design', 'qualification', 'IV&V', 'V&V', 'verification', or 'process' within the same sentence/window; or (b) split the concept into 'Process-level V&V' (qualification testing, IV&V, V&V, system verification, design verification, validation plan, qualification test) vs. 'General verify/validate verbs' (which probably should be dropped from process_scaffolding entirely as too generic)._
- `Scrum framework`: Two-letter acronym keywords (SM, PO) collide with unrelated SAP/Oracle/AWS terminology and add hundreds of false positives across the broader topic; CSM collides with ServiceNow Customer Service Management which is a strong false-positive cluster. → _Drop 'SM' and 'PO' outright. For 'CSM', require co-occurrence with 'scrum'/'agile'/'certified'/'master' in same sentence; otherwise the ServiceNow CSM module dominates._
- `Kanban / Lean / Waterfall and other methodologies`: 'SAFe' is duplicated here and in 'Agile methodology' (557 hits each, presumably the same hits), and 'LeSS' captures CSS preprocessor usage rather than Large-Scale Scrum. → _Deduplicate 'SAFe' across the two concepts (decide which concept owns it). Apply case-sensitive matching for both 'SAFe' and 'LeSS' with required context words ('framework', 'scrum', 'agile', 'release train')._
- `SDLC and process governance`: 'ART' (515 hits) and 'PLC' (110 hits) are three-letter tokens whose dominant meanings ('state of the art' and 'Programmable Logic Controller') are unrelated to process governance and inflate the concept by ~600 postings. → _Drop both. The intended SAFe Agile Release Train meaning is already captured by 'agile release train' / 'release train' / 'release trains'. The Product Life Cycle meaning is already captured by 'product lifecycle' / 'product life cycle'._
- `Project / program management roles & tooling`: Two- and three-letter role acronyms (PM, TPM, TPMs) carry severe semantic collisions (clock time, Trusted Platform Module, Technical Performance Measures) and 'Linear' as a tool name collides with 'linear algebra' which dominates DS/ML postings. → _Drop 'PM', 'TPM', 'TPMs', 'Linear'. Spelled-out forms ('project manager', 'project managers', 'technical program manager', 'technical program managers', 'PMs') already cover the role meanings cleanly._

### Legacy-stack markers (`legacy_stack`)

The vocabulary is broadly well-targeted at legacy enterprise stacks but suffers from several short-token false positives that fire on unrelated text: 'RPG' matches video-game RPG, 'IMS' matches the staffing firm 'IMS' and 'CMS/IMS', 'WAS' matches the past-tense English verb almost universally, 'VMS' matches generic plural 'VMs'/'cloud VMs' rather than OpenVMS, 'CVS' matches the pharmacy/retailer CVS, 'Hudson' matches place names (New Hudson, Hudson River Trading), and 'Ant' matches the proper noun ANT in deployment toolchains plus 'Salesforce ant' which is itself fine but 'Ant' on its own is too noisy without word boundaries. Several keywords are not actually 'legacy' by 2024 standards and contaminate the topic: 'SOAP', 'WSDL', 'XSD', 'SOA', 'Service Oriented Architecture', 'LDAP', 'Active Directory', 'Kerberos', 'MuleSoft', 'Group Policy', 'Perforce', 'Ant', 'PL/SQL', 'Teradata', 'DB2', 'Hyper-V', and '.NET Framework' all fire heavily in modern hybrid/cloud JDs and on Salesforce/ServiceNow/AWS roles. The most reliable concepts are 'Mainframe languages and platforms' (COBOL/JCL/CICS/VSAM/REXX cluster cleanly), 'Legacy Microsoft .NET stack' specific markers (WebForms/WCF/VB.NET/Classic ASP/WPF), and 'Legacy Java EE / app server stack' core (J2EE/EJB/WebLogic/WebSphere/JBoss/Struts/JSP). New legacy mainframe markers visible in snippets but missing from the vocabulary include IDMS, ENDEVOR, TSO/ISPF, File Aid, Xpediter, Abend Aid, ACOB/UCOB, CLIST, and IBM ECM/Content Navigator; Oracle Forms variants and Oracle ADF/Webcenter are also undercaptured. Recommended structural fix: split the SOAP/SOA concept into a 'mainframe-and-truly-legacy SOAP' subset versus 'still-current REST+SOAP integration' which should be excluded, and move modern items (MuleSoft, Active Directory, LDAP, Kerberos, Group Policy) out of legacy-stack into integration/identity topics.

_17 drops · 14 guards · 18 adds · 7 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Legacy Java EE / app server stack` — 3.6%
- `Legacy database platforms` — 3.3%
- `Legacy enterprise SOAP / web services` — 3.1%
- `Legacy enterprise integration / ESB / messaging` — 2.7%
- `Legacy Microsoft .NET stack` — 2.4%
- `Legacy Microsoft server / collaboration` — 2.3%
- `Legacy virtualization and on-prem infrastructure` — 2.0%
- `Legacy identity and directory services` — 1.5%
- `Mainframe languages and platforms` — 1.4%
- `Legacy version control and build` — 1.1%
- `Legacy enterprise applications and ERPs` — 0.7%
- `Legacy general-purpose languages` — 0.6%

**Top guards** (false-positive risks worth fixing):
- `MVS` (Mainframe languages and platforms): Possible 3-letter collisions, but sampled snippets are clean ('Mvs Cobol', 'MVS JCL', 'CICS, MVS, ISPF'). One snippet 'MVS, VIP, RSI' is ambiguous. → _guard_: Require co-occurrence with mainframe context tokens (z/OS, JCL, COBOL, CICS, ISPF, OS/390) within +/-50 chars to be safe; otherwise it's mostly fine.
- `ESX` (Legacy virtualization and on-prem infrastructure): 3-letter token. Sampled hits are all VMware ESX, but 'ESX' could match ESXi tail or proper-noun strings. → _guard_: Require word boundary and either 'VMware' within +/-30 chars or membership in a hypervisor list; current data is clean but the surface area is risky.
- `AIX` (Legacy virtualization and on-prem infrastructure): Risk of matching as a substring (titles like 'Software Engineer I, AIX'); current sample is clean and refers to IBM AIX UNIX. → _guard_: Word-boundary regex \bAIX\b only; keep.
- `Solaris` (Legacy virtualization and on-prem infrastructure): Clean in sample; sometimes appears in 'Oracle Solaris (UNIX)' lists alongside Linux/Windows in security-clearance JDs as a legacy-friendly skill rather than current focus. → _guard_: Keep as-is; high precision.
- `TFS` (Legacy Microsoft server / collaboration): Clean; TFS is unambiguously Team Foundation Server. Note many snippets describe migrating away from TFS to Azure DevOps — which is itself a legacy-stack signal. → _guard_: Keep; consider weighting 'migrating TFS' contexts as strong legacy-maintenance evidence.

**Top suggested additions** (grounded in corpus snippets):
- `IDMS` → Mainframe languages and platforms
- `CA-IDMS` → Mainframe languages and platforms
- `ISPF` → Mainframe languages and platforms
- `TSO` → Mainframe languages and platforms
- `TSO/ISPF` → Mainframe languages and platforms

**Concept-level recommendations:**
- `Legacy enterprise SOAP / web services`: The concept conflates 'a JD that mentions SOAP' (overwhelmingly modern integration roles requiring REST AND SOAP) with 'pre-REST SOAP-only legacy stack'. Hit rate of 3% on SOAP alone wildly overstates legacy presence. → _Reframe as 'pre-REST SOAP-only stack' and require absence of REST/RESTful/JSON-API in the same JD, OR require co-occurrence of >=2 SOAP-era markers (e.g., WSDL+CXF, JAX-WS+Axis, BPEL+OSB). Drop bare 'SOAP', 'WSDL', 'XSD' as standalone signals._
- `Legacy enterprise integration / ESB / messaging`: Mixes truly legacy ESBs (TIBCO BW, webMethods, BizTalk, Oracle SOA Suite, Mule ESB community edition) with modern iPaaS/cloud (MuleSoft Anypoint, current Kafka-adjacent ESB framings) and with the architectural pattern 'SOA' itself. → _Drop 'SOA' and 'Service Oriented Architecture' as keywords (they describe a pattern, not a legacy product). Move 'MuleSoft' to a separate 'integration platforms' topic. Keep TIBCO/webMethods/BizTalk/IBM MQ/MQSeries as legacy._
- `Legacy identity and directory services`: Active Directory, LDAP, Kerberos, Group Policy are all current technologies in hybrid-cloud and security-clearance JDs. Treating them as legacy markers contaminates the topic with cloud-architect and modern-IAM roles. → _Restrict the concept to clearly-deprecated identity products (Tivoli, SiteMinder, ADFS-on-prem). Move AD/LDAP/Kerberos/Group Policy to a 'hybrid-cloud / on-prem identity' topic, not 'legacy'._
- `Legacy database platforms`: DB2, Teradata, PL/SQL, Sybase ASE are still actively shipped/supported and appear in modern cloud-DW JDs. PL/SQL alone is one of the most-used DB skills today. → _Define 'legacy' more strictly: require version-pinning (DB2 for z/OS, Oracle <=11g, SQL Server <=2008, Sybase ASE <=15) or co-occurrence with a mainframe/on-prem context. Drop bare PL/SQL as a legacy marker._
- `Legacy version control and build`: Includes Perforce (current dominant VCS in game dev), Hudson (often a generic Jenkins-alternative mention), Ant (current Java build tool), and CVS (frequent collisions with CVS Health). The concept's hit rate is inflated by FPs. → _Restrict to ClearCase, PVCS, Visual SourceSafe/VSS, TFVC, MKS, StarTeam — pre-Git enterprise VCS. Drop Perforce and Ant entirely; keep CVS only with surrounding-context guard (within X chars of 'svn'/'subversion'/'version control'). GNU Make and Apache Ant are not legacy._

### Context infrastructure (`context_infrastructure`)

The vocabulary is dominated by a few generic anchor terms that drive most of the topic's hit rate but capture phenomena other than 'context infrastructure': 'monitoring' / 'monitor' / 'metric(s)' / 'logging' / 'logs' fire on generic ops/security/business-metric language, and 'cross-functional' / 'cross-functional teams' / 'business requirements' / 'business needs' / 'written communication' fire on boilerplate filler that any posting contains. Several keywords are flatly wrong-concept ('observable' = adjective for systems, not telemetry; 'spans' = verb meaning 'covers'; 'notion' = the noun, not the SaaS; 'design reviews'/'design review' = code/electrical/security reviews, not ADR/RFC artifacts; 'ERD' = data-modeling diagrams, not architecture decision records; 'SOP'/'SOPs' = manufacturing/QA procedures, not engineering runbooks; 'business value'/'KPI'/'business outcomes'/'business impact' = generic motivational language, not product literacy; 'product knowledge' = product-line domain knowledge of what the company sells, not product-sense; 'developer experience' = years of dev experience as often as the DX discipline; 'DX' = Salesforce DX in the vast majority of hits). The Architecture-decision-records concept and the Runbooks concept are particularly contaminated by their dominant keyword: drop 'design reviews'/'design review' and 'operational excellence' and the concepts collapse to honest, low-frequency ADR/runbook signal. There are a few real omissions in the snippets ('golden signals', 'golden path', 'service catalog', 'incident review', 'distributed tracing' co-occurring with 'three pillars of observability', 'BRD', 'business requirements document', 'C4 diagrams') but most adds are modest. Overall the topic needs aggressive guarding/dropping of about 25-30 keywords before its hit rates can be trusted as a context-infrastructure signal rather than a mixture of ops, security, manufacturing-QA, and HR-boilerplate signals.

_23 drops · 14 guards · 12 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Observability & telemetry stack` — 34.2%
- `Cross-functional communication & coordination` — 25.1%
- `Product & business literacy` — 21.1%
- `Data-pipeline & data-integration hygiene` — 20.7%
- `Technical documentation authoring & maintenance` — 11.1%
- `Technical writing craft` — 9.8%
- `Architecture decision records & RFCs` — 6.5%
- `Runbooks, playbooks, & operational docs` — 4.8%
- `System understanding & internal knowledge sharing` — 3.6%
- `API & interface documentation` — 2.8%
- `Service-level reliability targets` — 2.4%

**Top guards** (false-positive risks worth fixing):
- `monitoring` (Observability & telemetry stack): Generic verb across security ('security monitoring'), sysadmin patch management, M&S team activity reports, business case monitoring. → _guard_: Require co-occurrence with one of: alerting, logging, dashboards, observability, telemetry, metrics, traces, Prometheus, Datadog, Grafana, Splunk, ELK, New Relic, CloudWatch, SLO, OpenTelemetry — or limit to compounds 'application monitoring', 'service monitoring', 'system monitoring'.
- `metrics` (Observability & telemetry stack): Hits on 'business metrics like CSAT, NPS, NDR', 'quality metrics', 'reliability and performance metrics' — captures product/QA metrics, not telemetry. → _guard_: Require co-occurrence with logs/traces/dashboards/observability/alerts/Prometheus/Datadog/Grafana, OR exclude when preceded by 'business', 'quality', 'KPI', 'success'.
- `logs` (Observability & telemetry stack): Generic ('inspecting application logs', 'logs and reproduction steps in defect tracking'); also 'change logs' / 'audit logs' contexts. → _guard_: Require co-occurrence with metrics OR traces OR observability OR aggregation/centralized/structured.
- `logging` (Observability & telemetry stack): Often appears as a feature of CI/CD or generic 'best practices for logging, monitoring, error handling' boilerplate. → _guard_: Require either a tool (Datadog, Splunk, ELK, Loki) OR a qualifier (centralized, structured, distributed, aggregation) within the same window.
- `snowflake` (Data-pipeline & data-integration hygiene): Snowflake is a data warehouse product but also generic SQL skill demand — fires on resumes that just list it as a checkbox skill rather than discussing pipeline hygiene. → _guard_: Acceptable as-is for the data-warehousing concept, but document that this is a tool-mention proxy, not a hygiene/quality signal. Consider downweighting versus 'data quality' / 'data validation' for the hygiene interpretation.

**Top suggested additions** (grounded in corpus snippets):
- `golden signals` → Observability & telemetry stack
- `three pillars of observability` → Observability & telemetry stack
- `incident management` → Runbooks, playbooks, & operational docs
- `on-call` → Runbooks, playbooks, & operational docs
- `BRD` → Architecture decision records & RFCs

**Concept-level recommendations:**
- `Architecture decision records & RFCs`: Concept hit-rate (0.065) is dominated by 'design reviews'/'design review'/'technical specifications' which capture electrical-design reviews, security reviews, and functional/technical-spec writing for waterfall delivery — not the deliberative ADR/RFC artifact this concept is meant to measure. Once those three are removed/guarded, the concept is genuinely rare (<1%). → _Either (a) accept that ADR/RFC culture is rare in this dataset and rely only on tight terms (ADR, ADRs, RFC, RFCs, architecture decision record(s), request for comments, design proposals, technical proposal(s), one-pagers, decision logs, RFC processes) and report low rate honestly, or (b) split into two sub-concepts: 'ADR/RFC artifacts' (tight) vs 'design-review activity' (broader, but understood to overlap with code/security/electrical review)._
- `Runbooks, playbooks, & operational docs`: Dominant keyword 'operational excellence' is corporate values-language; 'SOP'/'SOPs'/'standard operating procedure(s)' fire mostly in manufacturing/QMS/compliance contexts (ISO 9001, 21CFR820). Net effect: concept rate is inflated and mixes engineering runbooks with manufacturing-QA SOPs. → _Drop 'operational excellence', SOP family. Keep tight engineering-runbook terms ('runbook(s)', 'playbook(s)', 'postmortem(s)/post-mortem(s)', 'incident report(s)', 'on-call', 'incident management'). Expect concept rate to drop to ~2-3%, which is honest._
- `Product & business literacy`: Concept rate (0.211) is mostly 'business requirements' (n=1371) and 'business needs' (n=1070), which are generic SWE-posting boilerplate. Stripping them leaves a much smaller, more meaningful product/customer-literacy signal. → _Drop or hard-guard the four 'business <noun>' generics (requirements, needs, value, outcomes, impact). Promote tighter signals: 'product sense', 'product mindset', 'product thinking', 'product intuition', 'customer empathy', 'user research', 'domain expertise/knowledge', 'KPIs', 'OKRs'. Expected rate after cleanup: ~6-8%._
- `Cross-functional communication & coordination`: 'cross-functional' and 'cross-functional teams' alone account for ~25% of postings — the topic essentially measures whether the boilerplate phrase is present, not whether real coordination work is described. → _Either weight the bare 'cross-functional' lower than the function-named variants ('partner with product/design/PM', 'work with product', 'translate technical concepts'), or require the bare phrase to co-occur with a named non-engineering function within ±15 tokens to count toward the concept._
- `Technical writing craft`: 'written communication' (n=1123) is generic resume boilerplate ('excellent verbal and written communication') — does not measure technical-writing craft as defined. → _Hard-guard 'written communication' to require either 'technical' qualifier, 'documentation' nearby, or a written artifact noun. Concept rate after guard will drop to ~2-3%, which is the honest signal._

## Cross-list reconciliation

`collisions.json` lists every keyword appearing in ≥2 topics. The patterns below cluster the most common collisions and propose a canonical home for each.

**Top topic-pair collision counts:**

| Topic A | Topic B | # collisions |
|---|---|---:|
| context_infrastructure | verification | 38 |
| mentorship | people_management | 18 |
| orchestration | process_scaffolding | 17 |
| context_infrastructure | orchestration | 12 |
| context_infrastructure | process_scaffolding | 12 |
| process_scaffolding | verification | 12 |
| performance | verification | 11 |
| mentorship | verification | 10 |
| orchestration | verification | 9 |
| context_infrastructure | mentorship | 8 |

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
