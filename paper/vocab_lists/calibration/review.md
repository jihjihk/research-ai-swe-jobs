# Vocab Lists — Calibration Review

_Generated 2026-05-06T01:33:11+00:00._


Calibration sample: **19,433 SWE postings** stratified across `kaggle_arshkon/2024-04`, `kaggle_asaniczka/2024-01`, `scraped/2026-04`, matched against `description_core_llm`.


## Executive summary

- Input vocabulary: **2,254 keywords** across **88 core concepts** in **8 topics**.
- Recommended **drops:** 120 keywords (~5% of input — mostly zero-hit entries).
- Recommended **guards:** 121 keywords with high false-positive risk visible in example matches.
- Recommended **additions:** 91 new keywords surfaced from corpus inspection.
- **Cross-list collisions:** 129 keywords appear in 2+ topics; 6 reconciliation rules proposed below.

### Per-topic state

| Topic | Concepts | Keywords | Zero-hit | Drop | Guard | Add | Concept hit-rate range |
|---|---:|---:|---:|---:|---:|---:|---|
| **people_management** | 10 | 101 | 2 | 22 | 14 | 12 | 0.0% — 6.0% |
| **orchestration** | 10 | 193 | 4 | 2 | 17 | 16 | 0.1% — 63.3% |
| **verification** | 9 | 356 | 1 | 6 | 18 | 17 | 3.6% — 39.5% |
| **mentorship** | 11 | 210 | 4 | 26 | 18 | 6 | 0.1% — 21.8% |
| **performance** | 14 | 450 | 118 | 35 | 14 | 12 | 0.3% — 10.4% |
| **process_scaffolding** | 11 | 341 | 0 | 9 | 11 | 8 | 8.2% — 52.4% |
| **legacy_stack** | 12 | 232 | 1 | 6 | 13 | 7 | 0.2% — 3.6% |
| **context_infrastructure** | 11 | 371 | 0 | 14 | 16 | 13 | 1.8% — 36.2% |

## Per-topic findings

### People-management markers (`people_management`)

The list confuses three distinct phenomena: (a) the candidate IS a people manager, (b) the candidate reports to a manager, and (c) the candidate provides informal/technical mentorship — and the topic definition explicitly excludes (b) and (c). The largest-hit concept (Coaching/mentoring at 6.0%) is dominated by 'mentor junior' (n=551) and 'coaching' (n=333), which the topic definition itself flags as out-of-scope informal mentorship. Several high-hit role keywords ('Senior Manager' n=65, 'team leads' n=54, 'team leadership' n=123) fire predominantly when the term refers to the manager the candidate REPORTS to, to a peer the candidate COLLABORATES with, or to a generic soft skill. The 'Headcount and hiring authority' concept has many semantically vague phrases ('build a team', 'grow the team', 'expand the team') that fire on growth-narrative boilerplate rather than hiring authority. The strongest signal patterns observed in snippets — 'direct reports' as a noun phrase, 'line management', 'manage staff augmentation/onboarding/firing', 'compensation decisions/planning', 'people leader' style titles — are well-covered, but the list needs guards against possessive-flips ('reports to', 'reporting to a Senior Manager') and benefit-language ('your career growth', 'opportunities for career development').

_22 drops · 14 guards · 12 adds · 5 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Coaching, mentoring, and career development` — 6.0%
- `People management role` — 2.3%
- `Team leadership` — 1.5%
- `Team-size signals` — 1.3%
- `Supervisory verbs` — 0.8%
- `Performance management` — 0.3%
- `Headcount and hiring authority` — 0.3%
- `1:1 meetings` — 0.0%
- `Direct reports` — 0.0%
- `Termination, hiring/firing authority` — 0.0%

**Top guards** (false-positive risks worth fixing):
- `engineering manager` (People management role): Used as the role the candidate WORKS WITH or REPORTS TO ('Work with the Engineering Manager to establish...', 'reporting to the Engineering Manager'), or as a hybrid alternative ('whether as a Tech Lead, Architect, or hands-on Engineering Manager'). → _guard_: Require the term to appear in a posting where the title or role descriptor is itself 'Engineering Manager' (title match), or where the phrase is 'as an Engineering Manager' / 'Engineering Manager role' / 'we are seeking an Engineering Manager'. Exclude 'reports to <X> Engineering Manager' and 'work with the Engineering Manager'.
- `leading a team` (Team leadership): Often used in retrospective consultant CV descriptions ('experience leading a team') with no indication the posting itself involves people management; also appears in technical-lead-only contexts. → _guard_: Pair with people-mgmt corroborator within +/- 100 chars (e.g., 'reports', 'manage', 'hire', 'performance review') OR restrict to forward-looking phrasing in responsibility lists.
- `leading the team` (Team leadership): Frequently denotes technical SME role ('SME and leading the team by example', 'leading the team strategy', 'leading the team through transitions') without people-mgmt responsibilities. → _guard_: Require corroborating people-mgmt token (direct report, hire, performance, 1:1, headcount) in the same posting.
- `leads a team` (Team leadership): Often technical leadership without people-mgmt scope ('leads a team to write, deploy, and maintain software'). → _guard_: Require co-occurrence with at least one Performance management or Direct reports concept hit in the same posting.
- `leads the team` (Team leadership): Same technical-leadership pattern ('Leads the team to write, deploy, and maintain software', 'leads the team in code/design reviews'). → _guard_: Same corroborator requirement as 'leads a team'.

**Top suggested additions** (grounded in corpus snippets):
- `line management` → People management role
- `direct reports` → Direct reports
- `managing 3-5 engineers` → Team-size signals
- `compensation planning` → Performance management
- `compensation decisions` → Performance management

**Concept-level recommendations:**
- `Coaching, mentoring, and career development`: This concept is the topic's largest hitter (6.0% hit rate) but its top three keywords ('mentor junior', 'mentor engineers', 'mentoring engineers', 'coaching') overwhelmingly fire on senior-IC postings where mentorship is described as a peer/IC contribution. The TOPIC definition explicitly excludes 'informal mentorship' yet the CONCEPT's own definition undermines that exclusion by including 'mentoring' as a primary semantic anchor. This is the structural reason the topic's apparent prevalence is inflated. → _Either (a) split this concept into two — 'manager-led career development' (career conversations, performance feedback, individual development plans, talent development of subordinates) and 'IC mentorship' (which should then be excluded from the topic), or (b) tighten this concept to require an explicit subordinate-relationship token ('your team', 'direct reports', 'engineers reporting to you') within the same sentence as the mentor/coach verb. Drop the bare 'coaching' and 'mentor junior' family unless guarded._
- `Team leadership`: Conflates (a) being a tech lead (technical leadership only), (b) being a people-managing team lead, and (c) collaborating with peer team leads. The verbs 'lead a team / leading a team' surface all three meanings indistinguishably. → _Require all 'lead/leading/leads + team' patterns to co-occur with at least one corroborator from {Direct reports, 1:1 meetings, Performance management, Headcount, Termination} concepts within the same posting. Alternatively, demote this concept to a 'weak signal' tier and only count when an additional people-mgmt concept is also present._
- `Headcount and hiring authority`: Mixes high-specificity hiring-authority phrases ('hire and onboard', 'interview loops', 'recruiting and interviewing', 'lead hiring') with low-specificity team-growth narratives ('build/grow/scale/expand the team') that fire on startup pitch boilerplate. The latter group inflates concept hit rate without indicating actual authority. → _Separate into two sub-concepts: 'Hiring authority (specific)' — recruiting, interview loops, hiring decisions, onboarding, hire & onboard, talent acquisition — and 'Team growth narrative (weak)' — build/grow/scale/expand the team. Only count the former for the people-management indicator; treat the latter as auxiliary._
- `Direct reports`: All four current keywords are second-person verb phrases ('will report to you', 'reporting to you', 'report into you', 'managing direct reports'), missing the most common noun phrase 'direct reports' alone, which appears in many real management postings and is the strongest single-token signal. → _Add the bare noun phrase 'direct reports' (with optional possessive variants like 'your direct reports', 'their direct reports'). Also add 'N direct reports' / 'direct report' singular._
- `Termination, hiring/firing authority`: Concept has only 3 keywords with combined hit rate near zero; one ('staffing decisions') is a false positive and another ('disciplinary action') is dominated by compliance boilerplate. The strongest authentic phrasing observed is 'hire, retain... fire' embedded in a longer responsibility list. → _Drop 'staffing decisions' and guard 'disciplinary action' (require co-occurrence with people-mgmt terms in same sentence). Add 'hire, retain', 'administering disciplinary action', and 'firing decisions' as more specific replacements. Acknowledge that this concept may have inherently low recall and serve as a precision-only signal._

### Orchestration (`orchestration`)

The vocabulary's classical-orchestration concepts (Architecture, Task Decomposition, Workflow/Pipeline) are well-formed but overweighted by a few high-recall stems that double-count related concepts: 'architecture/architectural/architectures' alone produce >70% of the topic's volume, with 'architectures' frequently denoting neural-network/ML model architectures and 'orchestration' frequently denoting Docker/Kubernetes container orchestration. The Repository-Instructions concept is broken: 'skills' (12,294 hits, 63%) is essentially a pure false positive matching the universal 'skills' English word in 'communication skills', 'analytical skills', 'job skills' etc., and dominates the entire concept. Several agent/AI keywords collide with non-AI senses ('A2A' = application-to-application integration, 'function calling' = Python async function calling, 'red-teaming' = security red-teaming, 'multi-agent' / 'multi-agent coordination' = robotics/UAV multi-agent systems, 'checkpointing' = ML training checkpointing, 'session state' = generic web app state, 'RFCs' = ITIL Request for Change). The new agent-stack concepts (Context Engineering, Agent Harnesses, Multi-Agent, Memory, Eval/Steering) are otherwise on target with strong precision on AI-context snippets. Several genuinely-used tools and patterns are missing from the list: 'AGENTS.md', 'Aider', 'Cline', 'Roo Code', 'Foundry agents', 'Kiro', 'LangSmith', 'Promptfoo', 'tool-use', 'tool use', 'agent skills', 'task-state', 'agent communication', 'self-correction', 'Open Code', 'Agentforce', 'AIP'.

_2 drops · 17 guards · 16 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Repository Instructions and Agent Configuration Files` — 63.3%
- `Software Architecture and System Design` — 52.1%
- `Workflow Design and Pipeline Orchestration` — 31.8%
- `Agent Harnesses and Coding Agents` — 5.6%
- `Task Decomposition and Planning` — 3.0%
- `Context Engineering and Prompt/Spec Authorship for Agents` — 2.6%
- `Multi-Agent Coordination` — 1.2%
- `AI Tool and Agent Evaluation/Steering` — 0.8%
- `Specifications and Requirements Authoring` — 0.6%
- `Agent Memory and State Management` — 0.1%

**Top guards** (false-positive risks worth fixing):
- `skills` (Repository Instructions and Agent Configuration Files): Plain English noun: 'communication skills', 'analytical skills', 'Job Skills', 'soft skills', 'problem-solving skills'. → _guard_: If kept at all, require co-occurrence with 'agent', 'agentic', 'Claude', 'sub-agent', 'tool', or appearance inside a list of agent primitives (e.g., 'skills, prompts, and subagents') within a 60-char window. Otherwise DROP (preferred).
- `architectures` (Software Architecture and System Design): ML/neural-network sense: 'machine learning architectures', 'transformer architectures', 'RAG architectures', 'data architectures', 'LLM architectures'. → _guard_: Exclude when preceded by 'machine learning', 'neural', 'model', 'transformer', 'LLM', 'RAG', 'data', 'reference' within a 4-token window; or require co-occurrence with classical software-architecture markers ('microservice', 'service', 'system', 'enterprise', 'distributed', 'event-driven', 'cloud').
- `orchestration` (Workflow Design and Pipeline Orchestration): Container orchestration: 'containerization and orchestration technologies (Docker, Kubernetes)', 'Kubernetes orchestration'. While arguably workflow-related, this is DevOps/infra, not workflow-design orchestration. → _guard_: Exclude when within 30 chars of 'Docker', 'Kubernetes', 'container', 'k8s', 'EKS', 'AKS', 'GKE'. Keep when paired with 'workflow', 'pipeline', 'data', 'ML', 'agent', 'tool', 'process', 'job', 'task'.
- `workflow` (Workflow Design and Pipeline Orchestration): ServiceNow / dev-tooling workflow: 'ServiceNow Workflow development', 'Workflow / AWE' (PeopleSoft), 'modern frontend build tools and workflows', generic 'development workflow'. → _guard_: Exclude when preceded by 'frontend', 'build tools and', 'development', 'agile', or within 30 chars of 'ServiceNow', 'PeopleSoft', 'Jira', 'PUMs'. Keep with 'design', 'orchestrat', 'engine', 'automation', 'data', 'ML', 'agent'.
- `workflows` (Workflow Design and Pipeline Orchestration): Same ServiceNow / generic-dev-workflow false positives as 'workflow'; also 'ServiceNow forms and workflows', 'modern dev workflows', 'configure workflows'. → _guard_: Same guard as 'workflow' — exclude ServiceNow/PeopleSoft contexts and pure generic-dev-workflow uses; keep when paired with design/orchestration/automation/data/ML/agentic markers.

**Top suggested additions** (grounded in corpus snippets):
- `AGENTS.md` → Repository Instructions and Agent Configuration Files
- `tool use` → Agent Harnesses and Coding Agents
- `tool-use` → Agent Harnesses and Coding Agents
- `agent skills` → Repository Instructions and Agent Configuration Files
- `Aider` → Agent Harnesses and Coding Agents

**Concept-level recommendations:**
- `Repository Instructions and Agent Configuration Files`: Concept is dominated entirely by 'skills' (12,294/12,299 hits = 99.96%) which is a pure false positive. After dropping 'skills', the concept has only 5 keywords with a combined ~5 hits — operationally this concept will be near-zero coverage at the desired precision. The concept itself is well-defined and important (CLAUDE.md/AGENTS.md/.cursorrules are real artifacts) but the named-file vocabulary is genuinely rare in JD copy at this date. → _Drop 'skills'; add 'AGENTS.md' and 'agent skills' (per ADD list) to broaden recall slightly. Accept that this concept will be a low-volume 'leading indicator' signal (<0.1% hit rate) rather than a base-rate concept; report it as such in the paper. Optionally widen by adding 'system prompts and CLAUDE.md context' regex-style multiword expressions._
- `Software Architecture and System Design`: Concept hit-rate is 52% — driven almost entirely by 'architecture' (35.4%), 'architect' (13.6%), 'architectures' (12.9%). This concept will essentially be 'is this a senior SWE posting?' rather than a discriminating orchestration signal. 'architectures' specifically pulls in ML-model-architecture noise (~2,500 hits with high FP rate). → _Either (a) move bare 'architecture'/'architect' to a broader 'senior-engineer base-rate' indicator and keep only the more specific keywords ('architectural decisions', 'high-level design', 'system design', 'ADRs', 'design tradeoffs', 'architectural patterns', 'distributed systems design') in this concept, or (b) keep them but explicitly normalize the concept hit-rate by reporting it separately from per-concept dominance metrics. Apply the 'architectures' guard regardless._
- `Workflow Design and Pipeline Orchestration`: 'orchestration' alone (1,786 hits) is genuinely ambiguous between (a) container orchestration (Docker/K8s — adjacent to DevOps), (b) data/workflow orchestration, and (c) agent/tool orchestration. 'pipeline' alone is mostly CI/CD pipeline. Without guards, this concept conflates DevOps base-rate with the orchestration archetype the paper actually wants to measure. → _Apply the container-exclusion guard on 'orchestration' and CI/CD-exclusion guard on 'pipeline'. Consider promoting the more specific tool-named keywords (Airflow, Step Functions, Dagster, Prefect, Argo Workflows, Temporal-if-added) and de-emphasizing the bare stems when computing topic membership, so the concept becomes a 'workflow-engine fluency' signal rather than a generic-pipeline signal._
- `Multi-Agent Coordination`: 'multi-agent', 'multi-agent systems', 'multi-agent coordination' all have meaningful collision with robotics/RL/UAV multi-agent literature, which is a legitimate but distinct technical domain represented in defense/aerospace SWE postings (Anduril Fury, Mission Software Engineer family). Without guards, this concept overcounts robotics-SWE postings as 'agent orchestration'. → _Apply robotics-exclusion guards (per GUARD list) on the three 'multi-agent*' keywords. Optionally split the concept into 'LLM Multi-Agent Coordination' vs 'Multi-Agent Systems (robotics)' if the paper cares about distinguishing them; otherwise just guard._
- `Agent Memory and State Management`: The concept's strongest-hit keyword 'checkpointing' is entirely ML-training checkpointing (not agent memory). Without it, the concept has only ~30 total hits across 9 narrow keywords — usable as a leading-indicator but very low base rate. 'session state' also leaks into generic web-session usage. → _Drop 'checkpointing' (per DROPS). Add 'self-correction' and 'task state' (seen in 'task state, routing, retries, and safe boundaries') to broaden recall. Apply session-state guard. Accept low base-rate and treat as emergent-skill marker._

### Verification (`verification`)

Vocabulary list is broadly well-targeted but contains several high-volume polysemous tokens that need guards: 'X-Ray' frequently fires on AWS X-Ray (correct, but also Jira X-ray test management, cross-listing concept), 'guardrail(s)' fires on AI safety AND on military weapons systems and unrelated business 'guardrails', 'monitors'/'alerts' fire on Workday-style notifications and product-monitoring devices unrelated to observability, 'observable' fires on the adjective ('scalable, observable systems') which is on-topic but 'on-call' fires heavily on customer support shifts and 'on call' has frequent COC-style false-positive title hits. The 'Reproducibility and artifact proof' concept conflates true reproducibility (formal verification, audit trails, reproducible builds) with generic 'logs/logging' and 'artifacts' which are observability and SDLC outputs. 'audit', 'auditing', 'auditor(s)' frequently fire on quality auditors (ISO 9001), data-usage audits, and engineering reviews unrelated to compliance verification. Coverage of modern AI evals is solid; missing items include 'eval suite', 'eval set', 'shift left' (already 'shift-left' present), 'pre-commit hooks', 'release notes', 'change failure rate', 'MTTR', 'MTBF', 'incident' (singular), and 'runbooks' which appear in observability snippets.

_6 drops · 18 guards · 17 adds · 5 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `CI/CD and quality gates` — 39.5%
- `QA and validation` — 28.7%
- `Automated testing` — 24.2%
- `Post-deployment observability` — 23.8%
- `Compliance and security verification` — 16.3%
- `Code review` — 16.0%
- `Reproducibility and artifact proof` — 9.6%
- `Static analysis and code quality tooling` — 7.0%
- `Evaluations and AI output verification` — 3.6%

**Top guards** (false-positive risks worth fixing):
- `guardrails` (Evaluations and AI output verification): Fires on 'Guardrail Common Sensor (GRCS)' military system, 'final guardrail for tax liabilities' (financial control), 'metrics and guardrails' (analytics constraints) — non-AI usage. → _guard_: Require co-occurrence with AI/ML/LLM/GenAI/prompt/model context within ~50 chars, OR keep the AI-specific 'AI guardrails'/'LLM guardrails' phrasing only.
- `guardrail` (Evaluations and AI output verification): Same as above — singular form is the worst offender ('Guardrail Common Sensor', 'final guardrail for ledger'). → _guard_: Require AI/LLM/prompt/model/safety co-occurrence; alternatively drop the singular and keep only AI-qualified phrases.
- `ground truth` (Evaluations and AI output verification): Fires on geospatial intelligence ('ground truth testing of intelligence'), generic data set construction, mixed-reality data generation — uses unrelated to AI eval. → _guard_: Require co-occurrence with model/eval/LLM/AI/labels within ~80 chars; otherwise treat as ML data engineering.
- `benchmarks` (Evaluations and AI output verification): Fires on 'compensation benchmarked against local market', 'storage benchmarks', 'performance benchmarks' (generic perf testing) — not AI eval benchmarks. → _guard_: Require co-occurrence with model/LLM/AI/eval/dataset within ~50 chars.
- `benchmarked` (Evaluations and AI output verification): The single hit is 'compensation benchmarked against local market' — pure salary FP. → _guard_: Require model/eval/LLM context, or drop given low volume.

**Top suggested additions** (grounded in corpus snippets):
- `MTTR` → Post-deployment observability
- `runbooks` → Post-deployment observability
- `incident` → Post-deployment observability
- `observability` → Post-deployment observability
- `tracing` → Post-deployment observability

**Concept-level recommendations:**
- `Reproducibility and artifact proof`: Concept conflates three distinct things: (1) true reproducibility/formal-verification (formal verification, reproducible builds, reproducible research); (2) generic SDLC artifacts (test artifacts, code artifacts) which are deliverables not proof; (3) audit trails/logs which are compliance evidence. Generic 'log/logs/logging' and 'artifacts' dominate hit volume but are off-topic for the stated definition. → _Tighten to 'Reproducibility and formal proof' (keep: reproducibility, reproducible, reproducible build(s), reproduce, reproducing, reproducible research, formal verification, log evidence). Move 'audit trail(s)', 'audit log(s)' to Compliance and security verification. Move 'logging', 'logs', 'log', 'traces' to Post-deployment observability (where the snippets clearly point — observability pillars). Drop bare 'artifacts' as too generic; keep 'build artifact(s)' as the precise reproducibility variant._
- `Evaluations and AI output verification`: Mixes AI-specific eval keywords (LangSmith, LangFuse, RAGAS, evals) with general-purpose terms (benchmarks, ground truth, red team, human-in-the-loop) that fire heavily in non-AI security and ML contexts. → _Either (a) split into two sub-concepts: 'AI/LLM evals (tool/method specific)' vs 'human-in-the-loop and ground-truth verification (general ML)', or (b) add AI-context guards on the polysemous terms (benchmarks, ground truth, red team, guardrails) so they only fire when in AI/LLM/model context. The current single concept blurs traditional pentesting red-teams into AI red-teaming, inflating the AI-eval count._
- `Post-deployment observability`: Concept correctly captures the three observability pillars (metrics, logs, traces), incident response, and on-call, but several keywords are too generic ('alerts', 'monitors', 'observable', 'dashboards', 'metrics') and fire on Workday/business/general usage. Also missing the gerund 'monitoring' and noun 'observability' which are the canonical forms. → _Add 'monitoring', 'observability', 'tracing', 'MTTR', 'MTBF', 'runbooks', 'incident' (singular). Remove or guard 'monitors', 'alerts', 'observable' (adjective). Optionally split deployment-strategy keywords (canary, blue-green) into a separate 'Progressive delivery' sub-concept since they describe deployment risk-reduction rather than post-deployment observability proper._
- `Code review`: Concept is well-scoped but very small (18 keywords, hit_rate 0.16). The dominant 'code reviews' (12.4% hit rate) carries almost the whole signal; long tail is fine. → _No structural change; consider adding 'code review feedback' and 'PR feedback' for completeness. Note that 'review process' fires on non-code review processes (TLS handshake review process, IAC TAD review) — consider guarding with code/PR/git context._
- `QA and validation`: Bare 'verification' and 'validation' fire heavily in defense/aerospace systems engineering ('verification and validation', 'V&V') as well as in software QA. This is on-topic for the broader 'verification' construct but conflates SDLC verification with systems-engineering V&V. → _Acceptable for the topic as defined; if RQ requires distinguishing software QA from systems V&V, add a guard requiring 'test'/'QA'/'software' context within ~50 chars._

### Mentorship markers (`mentorship`)

The 'Direct mentorship language' and 'Pair / mob / ensemble programming' concepts are very clean and on-target; the rest of the topic suffers significant semantic drift. The biggest problem cluster is 'Code review as teaching' and 'Architecture and design guidance', where high-volume keywords like 'review code' (156), 'reviewing code' (37), 'constructive feedback' (155), 'design discussions' (209), 'peer review' (88), 'technical direction' (474), 'technical leadership' (1048), 'technical guidance' (518), and 'raise the bar' (70) overwhelmingly fire on generic seniority/QA-gate language with no teaching framing. 'Onboarding and ramp-up' is polluted by data/tool/equipment onboarding ('onboard datasets', 'on-board sensors', 'integrations onboarded'). 'Explicit learning culture' is dominated by candidate-trait phrases ('growth mindset', 'intellectual curiosity', 'continuous learning') that describe the applicant rather than the team's mentorship environment. 'Teaching and knowledge transfer' has a small set of high-FP terms ('teach', 'teaching') that match TA experience and presentation skills. Recommend strict drops for ambiguous singletons and concept-level redefinitions for code-review/architecture/learning-culture.

_26 drops · 18 guards · 6 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Direct mentorship language` — 21.8%
- `Guidance and influence` — 11.4%
- `Teaching and knowledge transfer` — 4.7%
- `Architecture and design guidance` — 4.3%
- `Coaching and developing engineers` — 4.1%
- `Code review as teaching` — 3.7%
- `Explicit learning culture` — 3.6%
- `Onboarding and ramp-up` — 2.5%
- `Pair / mob / ensemble programming` — 1.0%
- `Communication and feedback skills` — 0.3%
- `Debugging and code-reading as a craft` — 0.1%

**Top guards** (false-positive risks worth fixing):
- `coach` (Coaching and developing engineers): Fires on non-engineer audiences: 'coach team members' in IT/HR/Custom Apps context, 'Coach, train, and guide non-technical end users', 'coach business stakeholders'. → _guard_: Require co-occurrence with engineer-target token within ~10 tokens: engineer(s)|developer(s)|junior|peers|the team|other engineers|less experienced. Exclude when followed by 'customers'|'stakeholders'|'end users'|'non-technical'.
- `coaching` (Coaching and developing engineers): Frequently appears in 'coaching, training, and mentoring' generic phrasing about non-engineer staff or subordinates broadly. → _guard_: Same engineer-target proximity check as 'coach'; or require pairing with mentor|junior|engineers|developers|team members in same sentence.
- `onboard` (Onboarding and ramp-up): Fires on data/tool/integration onboarding: 'onboard new datasets', 'onboard integrations', 'onboard clients', 'onboard new experimental techniques'. → _guard_: Require human object: 'onboard (new |junior |)?engineers|developers|hires|colleagues|team members|interns|teammates'.
- `onboarding` (Onboarding and ramp-up): Fires on customer onboarding flows ('onboarding flow', 'employee lifecycle onboarding'), data onboarding, partner integration onboarding. → _guard_: Require proximity to engineer|developer|hire|new colleague|junior|new employee within ~6 tokens, OR pair with 'developer onboarding'|'engineer onboarding'|'new hire onboarding'.
- `on-board` (Onboarding and ramp-up): Fires on hardware contexts: 'on-board sensors', 'on-board the robot' (literal physical on-board). → _guard_: Exclude when adjacent to sensor|robot|aircraft|device|computer; require human object as with 'onboard'.

**Top suggested additions** (grounded in corpus snippets):
- `teach by example` → Teaching and knowledge transfer
- `teaches by example` → Teaching and knowledge transfer
- `teach informally` → Teaching and knowledge transfer
- `force multiplier` → Communication and feedback skills
- `level up the team` → Coaching and developing engineers

**Concept-level recommendations:**
- `Code review as teaching`: The concept is bleeding heavily into Verification semantics. The high-volume keywords ('review code', 'reviewing code', 'peer review', 'constructive feedback', 'design discussions', 'pr reviews', 'code-reviews') fire overwhelmingly on neutral correctness-gate language with no teaching/feedback/growth framing. Concept hit rate (3.68%) is dominated by these false-positive-prone signals; the genuinely teaching-framed phrases ('thoughtful code review', 'review that teaches', 'code review that teaches', 'deep-dive code reviews') hit only ~5 postings combined. → _Restrict the concept to phrases that explicitly couple review with teaching/feedback/growth language: keep 'thoughtful code review', 'review that teaches', 'code review that teaches', 'deep-dive code reviews', 'thoughtful feedback'. For the broad terms ('review code', 'peer review', 'code reviews'), require co-occurrence within the same sentence with mentor|teach|grow|junior|feedback|learn — implement as compound patterns rather than standalone keywords. Move the bare review terms to Verification._
- `Architecture and design guidance`: Mixes two distinct senior-IC behaviors: (a) shaping technical decisions for/with other engineers (mentorship-adjacent) and (b) plain architecture/strategy work (not mentorship). Keywords like 'technical direction' (474), 'establish best practices' (55), 'define best practices' (27), 'promote best practices' (47), 'drive architecture' (27), 'design docs' (34), 'shape the architecture' (8), 'define standards' (21) all describe what a senior IC does, not necessarily teaching/growing others. → _Split into two narrower clusters: (1) 'Architecture review/critique as guidance' — keep only phrases that imply review-of-others' work or co-design with others ('architectural guidance', 'architecture guidance', 'design guidance', 'architectural review(s)' when paired with mentoring); (2) drop the pure-strategy keywords ('technical direction', 'set technical direction', 'shape the architecture', 'drive design decisions', 'drive architecture', 'design docs', 'establish patterns') from mentorship — these belong in a separate Senior-IC-scope topic. Keep 'raise the technical bar', 'raise the engineering bar', 'raise engineering standards' only with engineer-target proximity guard._
- `Explicit learning culture`: Concept conflates team-culture statements ('we have a learning culture') with candidate-trait requirements ('you have a growth mindset'). Most high-volume hits ('continuous learning', 'growth mindset', 'intellectual curiosity', 'intellectually curious', 'lifelong learner') are candidate traits — they describe who the company wants to hire, not whether the team mentors. → _Restrict concept to keywords that are unambiguously team/company-as-subject: 'culture of learning', 'learning culture', 'high-trust', 'psychological safety', 'blameless postmortem', 'blameless post-mortem', 'learn from each other', 'foster growth' (already in Coaching), 'learn from mistakes'. Drop or move all candidate-trait keywords listed in DROPs. Optionally rename to 'Team learning culture statements' to enforce the framing._
- `Onboarding and ramp-up`: Bare verbs 'onboard', 'onboarding', 'on-board', 'onboarded' fire heavily on dataset/integration/equipment onboarding. The concept is conceptually narrow but the keywords are not selective. → _Replace bare verbs with compound forms that require human objects: 'onboard new engineers', 'onboard junior', 'onboard new hires', 'onboard new colleagues', 'developer onboarding', 'engineer onboarding', 'new-hire onboarding', 'engineering onboarding'. Apply guards to the bare verbs as documented above if they must be retained._
- `Debugging and code-reading as a craft`: Only 3 keywords and the concept hit rate is 0.14% — extremely thin. 'walkthroughs' alone is too generic; 'reading code' is also placed in 'Code review as teaching' (cross-list within the same topic). → _Either drop the concept entirely (it has minimal coverage and the listed keywords don't add unique signal beyond Code-review/Pair-programming) or expand it with concrete observed phrases like 'debug together', 'read the code and build prototypes' (observed under 'influence without authority'), 'walking through the code', 'pair on debugging'. If retained, deduplicate 'reading code' (currently listed under both this concept and 'Code review as teaching')._

### Performance & deep technical understanding (`performance`)

The performance vocabulary is broadly well-targeted: high-hit keywords like 'low-latency', 'distributed systems', 'profiling', 'multi-threaded', 'CUDA', and 'systems programming' fire on genuine deep-systems job ads, and rare-but-precise terms (eBPF, DPDK, RDMA, JVM internals, MLIR) cleanly identify low-level work. The major false-positive risks are (a) generic depth-claim phrases applied to non-performance topics ('strong knowledge of Page Object Model', 'deep knowledge of math/probability', 'solid grasp of 3NF'), (b) hardware tokens that fire on hardware-validation/firmware roles unrelated to perf (FPGA on test-bench JDs, DRAM on memory-controller test, Verilog on firmware), and (c) abbreviations that collide with other meanings ('P4' as a vendor proficiency level, 'BGP' as a routing protocol that hits networking SREs not internals, 'APM' as generic observability tooling rather than profiling, 'Raft' as a company name, 'codegen' as Swagger/GraphQL Codegen). Several keywords are pure scope-creep (performance improvement/improvements, optimize performance, identifying bottlenecks, scalable systems, massive scale, algorithm design, numerical methods) that fire on ordinary product/engineering work and dilute the topic's claim of 'low-level/internals depth that AI cannot substitute'. Roughly 118/450 keywords are zero-hit, and several strong concepts (networking_internals, distributed_systems_internals, compilers_and_runtimes) carry many never-fired tokens that should be pruned to keep the vocabulary defensible.

_35 drops · 14 guards · 12 adds · 9 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `depth_claim_language` — 10.4%
- `performance_optimization_general` — 9.2%
- `distributed_systems_internals` — 7.3%
- `latency_throughput_scale` — 7.1%
- `profiling_and_benchmarking` — 5.0%
- `concurrency_parallelism` — 4.2%
- `low_level_systems_programming` — 3.9%
- `scaling_and_efficiency` — 2.3%
- `hardware_aware_programming` — 2.2%
- `os_kernel_internals` — 1.2%
- `database_storage_internals` — 0.9%
- `networking_internals` — 0.8%
- `algorithmic_optimization` — 0.6%
- `compilers_and_runtimes` — 0.3%

**Top guards** (false-positive risks worth fixing):
- `P4` (networking_internals): Hits 'P4 - Expert' as a vendor/HR proficiency-level marker (e.g., 'AWS Cloud Services (P4 - Expert)') unrelated to the P4 packet-processing language. → _guard_: Require co-occurrence with networking context tokens (e.g., 'OpenFlow', 'BGP', 'XDP', 'eBPF', 'dataplane', 'switch', 'packet') within +/-50 chars; otherwise drop.
- `BGP` (networking_internals): Most hits are routing-protocol mentions in network-engineering / SRE JDs ('OSPF, ISIS, RIP', 'anycast, BGP'), which is operator-level networking, not networking-stack internals. → _guard_: Require co-occurrence with internals signals ('kernel bypass', 'XDP', 'eBPF', 'dataplane', 'DPDK', 'P4', 'packet processing') to claim internals; otherwise treat as ordinary networking proficiency, not internals.
- `APM` (profiling_and_benchmarking): Most APM hits refer to general application-performance-monitoring vendors (Datadog, Splunk, ELK, Dynatrace, Grafana, Prometheus) used by SREs/DevOps, not active profiling for perf engineering. → _guard_: Either drop, or require co-occurrence with 'profiling'/'profiler'/'flame graph'/'trace' to retain the perf-engineering interpretation.
- `P4` (profiling_and_benchmarking): Same vendor-proficiency-level ('P4 - Expert') and dataplane-language collision; also misplaced — it is a packet-processing language, not a profiling tool. → _guard_: Remove from profiling concept entirely; if kept anywhere, only in networking_internals with the guard above.
- `FPGA` (hardware_aware_programming): Many hits are hardware-validation/firmware roles ('FPGA debug, chip bring-up', 'DSP and/or FPGA') describing FPGA as a target board, not software-side hardware-aware optimization. → _guard_: Require co-occurrence with software-perf signals ('acceleration', 'optimization', 'kernel', 'CUDA', 'compute', 'inference') OR keep but mark as a weak signal; firmware/test bench mentions should not count.

**Top suggested additions** (grounded in corpus snippets):
- `Triton` → hardware_aware_programming
- `Triton kernels` → hardware_aware_programming
- `HIP` → hardware_aware_programming
- `DirectML` → hardware_aware_programming
- `perf` → profiling_and_benchmarking

**Concept-level recommendations:**
- `depth_claim_language`: Phrases like 'strong knowledge of', 'deep knowledge', 'deep expertise', 'thorough understanding of', 'solid grasp of' fire heavily (sums of hundreds) but are not specific to performance/internals depth — they fire on 'strong knowledge of Page Object Model', 'deep knowledge of math/probability', 'solid grasp of 3NF data modeling', 'thorough understanding of JavaScript/TypeScript'. As-is, this concept inflates the performance topic with unrelated job ads. → _Either (a) drop this concept from the performance topic and use it only as a pattern that must co-occur with another performance keyword in the same posting, or (b) restrict matches to depth-phrases whose object-of-the-preposition is itself a low-level term ('deep knowledge of <kernel|systems|distributed systems|GPU|CUDA|networking|compilers|...>'). The current 'concept_hit_rate=0.10' is misleading._
- `performance_optimization_general`: 'performance improvement(s)', 'performance gains', 'optimize performance', 'optimizing performance', 'identifying bottlenecks' fire on generic product/QA work and dilute the 'low-level perf demand' interpretation. → _Tighten to keywords that explicitly imply engineering-side perf work: 'performance tuning', 'performance-critical', 'performance-sensitive', 'performance bottlenecks', 'HPC', 'high performance computing', 'performance engineer'. Drop the generic improvement/optimize-performance variants or move them behind a co-occurrence guard with profiling/latency/throughput._
- `scaling_and_efficiency`: 'scalable systems', 'massive scale', 'resource utilization' are generic backend-engineering boilerplate that does not signal internals depth or AI-resistant skill. → _Drop 'scalable systems'; demote 'massive scale' / 'resource utilization' to weak signals or require co-occurrence with latency/throughput/efficiency-quantification language. Concept definition should clarify it is about 'efficiency under scale targets' not 'we operate at scale'._
- `hardware_aware_programming`: FPGA/Verilog/VHDL/DRAM frequently fire on hardware-validation / firmware / chip-design roles where the candidate is the hardware engineer — not a software engineer doing hardware-aware programming. This violates the topic's intent (software-side internals). → _Either split into two concepts ('software hardware-aware programming' vs 'hardware-design adjacent') OR add a co-occurrence requirement with software-perf tokens (CUDA, GPU programming, compute, inference, kernel, optimization)._
- `concurrency_parallelism`: 16/41 keywords are zero-hit and the high-hit ones include 'coroutines' which is dominated by Kotlin/Android usage that the concept definition explicitly excludes ('beyond standard async/await usage'). → _Prune zero-hit theoretical terms (memory barriers, happens-before, livelock, actor model, green threads, fibers, rwlock); add a guard to coroutines that requires non-mobile context or low-level cues._

### Process-scaffolding markers (`process_scaffolding`)

The list is largely well-formed but contains four catastrophic false-positive keywords driving most of the topic's noise: 'LeSS' (567 hits, almost entirely matching the CSS preprocessor 'LESS' or the word 'less' due to case-insensitive substring matching), 'SAFe' (557 hits, dominated by 'safe operation', 'safe deployment', 'safe and maintainable'), 'ART' (515 hits, almost all 'state of the art' or 'state-of-the-art'), and to a lesser extent 'governance' (1345 hits, heavily contaminated by 'data governance', 'AI/LLM governance', 'cloud governance'). 'Validation' and 'verification' as bare tokens are dominated by data validation, test verification, and identity verification rather than process-level V&V, and 'audit' is contaminated by security/compliance audits unrelated to process governance. 'Specifications' is the dominant Specification-authoring keyword but frequently means API/file format specifications (ELF specs, REST specs) rather than authoring practice. The Requirements concept is dominated by the bare 'requirements' (9514 hits, 49 percent of the sample) which is so generic it acts as a 'job listing has any requirements section' marker rather than requirements engineering. Several genuine gaps exist (refactor 'specifications' to require authorship verbs, add 'ceremonies', 'demos', 'showcases', 'IPM', 'sprint demos', 'agile ceremony').

_9 drops · 11 guards · 8 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Requirements engineering` — 52.4%
- `SDLC and process governance` — 34.0%
- `Agile methodology` — 31.2%
- `Scheduling and milestones` — 20.1%
- `Project / program management roles & tooling` — 17.9%
- `Project coordination` — 15.8%
- `Scrum framework` — 13.6%
- `Verification & Validation (V&V)` — 11.9%
- `Specification authoring` — 11.5%
- `Sprints and iterations` — 8.5%
- `Kanban / Lean / Waterfall and other methodologies` — 8.2%

**Top guards** (false-positive risks worth fixing):
- `governance` (SDLC and process governance): Heavy false positives from 'data governance', 'LLM model governance', 'cloud governance', 'data governance policies', 'governance architecture' (Apigee proxy) — these are domain-specific governance, not SDLC process governance. → _guard_: Replace bare 'governance' with bounded variants: 'IT governance', 'process governance', 'engineering governance', 'project governance', 'governance framework', 'governance model'. Keep 'governance' only when preceded by IT/engineering/project/process/SDLC.
- `validation` (Verification & Validation (V&V)): Fires on 'data validation', 'DB Validations', 'Tableau dashboard development and validation', 'form validation' — low-level test/data validation, not process V&V. → _guard_: Require co-occurrence with verification or a process modifier ('system validation', 'design validation', 'process validation', 'qualification validation'); drop the bare token or downweight it.
- `verification` (Verification & Validation (V&V)): Fires on 'data verification testing', 'verification of engineering requirements' (mixed), 'identity verification' contexts, generic test verification. → _guard_: Require pairing with validation or a process modifier ('system verification', 'design verification', 'requirements verification', 'qualification testing'); the 'verification and validation' / 'V&V' bigrams are clean.
- `audit` (SDLC and process governance): Snippets show 'security access rights audit', 'FDA 21 CFR audit', 'supplier audits and client audit processes' — these are compliance/security audits, not process-improvement audits. → _guard_: Bound to 'process audit', 'code audit', 'release audit', 'audit trail' (process-context), or require co-occurrence with 'process'/'release'/'change' nearby.
- `refinement` (Scrum framework): Fires on 'continuous refinement of software', 'refinement of standards/processes', 'execution and refinement of core infrastructure' — generic 'making something better' usage. → _guard_: Require backlog/story/sprint context: 'backlog refinement', 'story refinement', 'sprint refinement' (already in list); drop bare 'refinement'.

**Top suggested additions** (grounded in corpus snippets):
- `agile ceremonies` → Agile methodology
- `scrum ceremony` → Scrum framework
- `showcases` → Scrum framework
- `sprint demo` → Scrum framework
- `agile environment` → Agile methodology

**Concept-level recommendations:**
- `Requirements engineering`: Concept hit rate is 52.4 percent of the entire sample, driven almost entirely by the bare 'requirements' keyword (49 percent on its own). This makes the concept a near-universal indicator of 'job description has a Requirements section', which is uninformative. → _Drop bare 'requirements'; the concept then reflects actual requirements-engineering practice (gathering, eliciting, traceability, BRD/SRD/RTM, business/functional/system/user requirements, DOORS, JAMA, Polarion). Expected hit rate after fix should drop to roughly 10-15 percent and become a more meaningful signal._
- `Verification & Validation (V&V)`: Bare 'validation' (1692 hits) and 'verification' (816 hits) dominate the concept but are heavily contaminated by data validation, identity verification, and unit-test verification — none of which are the systems-engineering V&V process. → _Restrict the concept to the bigram and modifier forms ('verification and validation', 'V&V', 'IV&V', 'system verification', 'design verification', 'system validation', 'design validation', 'process validation', 'qualification testing', 'ASPICE'). Move bare validation/verification under guard-or-drop. Also clarify the definition to explicitly contrast with low-level test verification (which it already mentions)._
- `Specification authoring`: The bare 'specifications', 'specification', 'spec', 'specs', 'specify', 'specifying' keywords mix the artifact noun, the authoring verb, and colloquial usage; they do not consistently mean 'authoring a spec as a process activity'. → _Restrict to authored-artifact forms with modifiers: technical/functional/design/product/system/API specifications, software requirements specification, SRS, ADRs, RFCs, ERD, design document. Drop the bare verbs ('specify', 'specifying') and bare nouns ('spec', 'specs') unless the topic is meant to capture casual spec-talk._
- `Kanban / Lean / Waterfall and other methodologies`: Two of the three top-hit keywords are catastrophic false-positives: 'LeSS' (CSS preprocessor) and 'SAFe' (the word 'safe'). Removing both will collapse this concept's measured hit rate substantially but produce a more honest signal. → _Drop 'LeSS' entirely (the framework is rarely named in postings; if needed use 'Large-Scale Scrum' / 'LeSS framework' as bounded forms). Replace 'SAFe' with 'SAFe framework', 'SAFe certification', 'Scaled Agile Framework', and rely on the existing 'scaled agile' / 'SAFe Agilist' variants. Same applies to the duplicate 'SAFe' under Agile methodology._
- `SDLC and process governance`: Concept is muddled: 'governance' (1345) and 'continuous improvement' (1093) are very high but match many non-process domains (data governance, model governance, generic improvement); 'ART' (515) is a pure false positive. → _Drop 'ART'; guard 'governance' (require IT/process/engineering/project context); split 'continuous improvement' into Lean/Six-Sigma-context vs generic-improvement uses if granularity matters, otherwise accept the noise. Consider promoting CMMI/ITIL/ISO-9001/Six-Sigma to a separate 'Process maturity frameworks' sub-concept._

### Legacy-stack markers (`legacy_stack`)

The list is broadly accurate but suffers from three structural problems: (1) several keywords are duplicated across concepts within this same topic (VSS, TFVC, BizTalk/BizTalk Server, JAX-WS, JAX-RPC) which double-counts postings and muddles concept attribution; (2) high-volume terms whose modern usage dominates the legacy meaning are silently inflating the topic rate (Active Directory is overwhelmingly Azure AD/Entra in cloud-architect JDs; LDAP/Kerberos/Group Policy commonly co-occur with modern SSO/Okta; SOAP web services and J2EE are routinely listed as one of many integration patterns next to REST/Spring Boot in greenfield JDs; PL/SQL is a current Oracle skill, not a legacy marker; VBA hits are 80%+ Excel macros, not legacy code maintenance; CVS hits include the pharmacy chain); (3) a handful of generic acronyms (IMS, MVS, ESB, OSB, IIB, NTLM, GPO, ESX) carry real polysemy and need keyword guards even though their dominant sense is correct. On the positive side, the core mainframe concept (COBOL/JCL/CICS/VSAM/REXX/IDMS) is well-covered with low FP rate, and several missing legacy markers visible in snippets should be added: Wildfly, MSMQ, Pacbase, RACF, ClearQuest, XenDesktop, Expediter. No keywords have a true zero-hit problem (only Visual SourceSafe at 0, but its abbreviation VSS captures the same postings).

_6 drops · 13 guards · 7 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Legacy Java EE / app server stack` — 3.6%
- `Legacy database platforms` — 3.3%
- `Legacy Microsoft server / collaboration` — 2.4%
- `Legacy version control and build` — 1.8%
- `Legacy Microsoft .NET stack` — 1.7%
- `Legacy virtualization and on-prem infrastructure` — 1.7%
- `Legacy identity and directory services` — 1.5%
- `Mainframe languages and platforms` — 1.5%
- `Legacy enterprise applications and ERPs` — 1.0%
- `Legacy enterprise integration / ESB / messaging` — 1.0%
- `Legacy general-purpose languages` — 0.6%
- `Legacy enterprise SOAP / web services` — 0.2%

**Top guards** (false-positive risks worth fixing):
- `Active Directory` (Legacy identity and directory services): Dominated by 'Azure Active Directory', 'Azure AD', 'Entra ID' mentions in modern cloud-architect JDs; the cloud variant (AAD/Entra) is the opposite of legacy. Several top examples show this: 'Configuring Azure Active Directory Sync', 'azure active directory, ztad, integration with MFA, Okta', 'Azure AD, LDAP, SSO, MFA'. → _guard_: Require the 4-token left context to NOT contain 'Azure', 'AAD', 'Entra', or 'cloud'; OR require co-occurrence in the JD with at least one of {on-prem, on premise, domain controller, GPO, Kerberos, LDAP, ADFS} to count as a legacy signal. Drop the bare 'Active Directory' match when 'Azure' appears within 3 tokens.
- `LDAP` (Legacy identity and directory services): Frequently appears alongside modern IAM (Okta, Ping Identity, ForgeRock, SSO/MFA, Azure AD) as a still-supported protocol, not as a legacy marker. LDAP is a current standard. → _guard_: Treat LDAP as legacy ONLY when co-occurring with on-prem markers (Active Directory non-cloud, OpenLDAP, eDirectory, Tivoli Directory) or when JD uses 'legacy/migrate/maintain'; otherwise downweight or exclude. Consider moving LDAP out of the indicator list entirely since it is a living protocol.
- `Kerberos` (Legacy identity and directory services): Same issue as LDAP — appears as one item in modern auth-protocol lists ('SAML, OAuth, and Kerberos', 'Hadoop ecosystem (Kerberos)', 'SAML, OAUTH, Kerberos, JWT Token, SSO'). Kerberos is still the standard for AD auth and Hadoop security. → _guard_: Require co-occurrence with on-prem AD signals (domain controller, GPO, ADFS, Group Policy) to count as legacy. Drop when listed alongside OAuth/JWT/SAML/OIDC in a generic auth-protocols enumeration.
- `Group Policy` (Legacy identity and directory services): Some hits are 'group policy' as a generic term ('STIG reviews, group policy and permissions') unrelated to AD GPOs. Other hits are valid (Domain Controller setup, GPO objects). → _guard_: Require capitalized 'Group Policy' OR 'GPO' co-occurring with 'Active Directory'/'Domain Controller'/'AD' within the same sentence; exclude standalone 'group policy' lowercase mentions about access policies.
- `GPO` (Legacy identity and directory services): Three-letter acronym; one example shows it grouped with 'AAD / Entra' (modern cloud), and another with 'AWS, and other modern deployment/infrastructure tools' — these are NOT legacy contexts. → _guard_: Word-boundary match only ('\bGPO\b'), AND require co-occurrence with 'Active Directory'/'AD'/'Group Policy' in the same JD; exclude when nearest neighbours include 'Entra'/'AAD'/'cloud'.

**Top suggested additions** (grounded in corpus snippets):
- `Wildfly` → Legacy Java EE / app server stack
- `MSMQ` → Legacy Microsoft .NET stack
- `Pacbase` → Legacy general-purpose languages
- `RACF` → Mainframe languages and platforms
- `ClearQuest` → Legacy version control and build

**Concept-level recommendations:**
- `Legacy Microsoft server / collaboration vs Legacy enterprise integration / ESB / messaging vs Legacy version control and build`: Three pairs of keywords (VSS, TFVC, BizTalk/BizTalk Server) are duplicated across concepts inside this single topic, causing the same posting to be counted under two concepts and inflating per-concept hit-rates. The 'Legacy Microsoft server / collaboration' concept currently mixes (a) actual collaboration servers (SharePoint, Lync, InfoPath, Exchange-era), (b) BI/data tools (SSIS, SSRS, SSAS, Crystal Reports), (c) version control (TFS, TFVC, VSS), and (d) integration (BizTalk). These are four distinct semantic clusters. → _Split 'Legacy Microsoft server / collaboration' into: (i) 'Legacy Microsoft collaboration' (SharePoint 2010-2016, Lync, Skype for Business, InfoPath, Exchange on-prem); (ii) 'Legacy Microsoft BI stack' (SSIS, SSRS, SSAS, Crystal Reports — note Crystal Reports is SAP, may belong elsewhere); (iii) move TFS/TFVC/VSS into 'Legacy version control and build' as the canonical home; (iv) move BizTalk/BizTalk Server into 'Legacy enterprise integration / ESB / messaging' as the canonical home. Remove the duplicate entries left behind._
- `Legacy enterprise SOAP / web services vs Legacy Java EE / app server stack`: JAX-WS and JAX-RPC are duplicated across both concepts. They are SOAP/WS-* APIs, not app-server features. → _Keep JAX-WS and JAX-RPC only in 'Legacy enterprise SOAP / web services'; remove them from 'Legacy Java EE / app server stack'._
- `Legacy identity and directory services`: Three of the four highest-volume keywords in this concept (Active Directory n=202, LDAP n=82, Kerberos n=28, Group Policy n=37) are NOT reliably 'legacy' — they are routinely used in modern hybrid-cloud and on-prem-AD environments, and Active Directory in particular is dominated by 'Azure Active Directory' / 'Entra ID' mentions, which are the OPPOSITE of legacy. The concept currently risks misclassifying ~300 modern cloud-IAM postings as legacy. → _Apply the per-keyword guards above (require on-prem context, exclude Azure/Entra/cloud co-occurrence). Consider redefining the concept as 'On-prem Windows-domain identity stack' and tightening to: ADFS, Tivoli Identity Manager, SiteMinder, OpenLDAP, NTLM (with guard), Kerberos+AD-only context. Move the bare 'LDAP' and 'Kerberos' tokens to a 'protocol' tier with low individual weight._
- `Legacy general-purpose languages — VBA placement`: VBA's 51 hits are overwhelmingly 'Excel VBA' for trading desks and finance teams — this is current Microsoft Office macro work, not maintenance of pre-2015 enterprise systems. Including it inflates the legacy-language signal with a population of Office-power-user developers who are not legacy maintainers. → _Either (a) split VBA out into a separate 'Office automation' tag and exclude from legacy_stack scoring, or (b) restrict VBA matches to those co-occurring with 'Visual Basic 6'/'VB6'/'Access database'/'legacy'._
- `Legacy database platforms — PL/SQL placement`: PL/SQL (n=268) is currently the second-highest-hit keyword in the topic, but PL/SQL is the active stored-procedure language for current Oracle Database (19c, 23c). Its presence in a JD does not signal 'legacy database platform' — it signals 'Oracle DB skill', which is mainstream today. → _Remove bare 'PL/SQL' from this concept, OR require co-occurrence with 'Oracle 8i'/'9i'/'10g'/'11g'/'Oracle Forms'/'Oracle Reports' to count as legacy. Without this, PL/SQL alone is a false-positive driver._

### Context infrastructure (`context_infrastructure`)

The vocabulary fires on 371 keywords with zero dead entries, but several high-volume terms drive most of the topic mass through false positives: 'monitoring' (3107 hits), 'metrics' (1476), 'stakeholders' (3994), and 'cross-functional' (3453) routinely match generic language that has nothing to do with observability or stakeholder coordination as defined. The 'Architecture decision records & RFCs' concept is silently absorbed by 'technical specifications' (419 hits) and 'design artifacts' (83), which in snippets overwhelmingly mean requirements-to-spec translation and generic design deliverables, not deliberative ADR/RFC artifacts. The 'Data-pipeline & data-integration hygiene' concept conflates hygiene/governance language (lineage, contracts, dictionaries) with raw tool names (snowflake, redshift, databricks, ETL) that signal data-engineering roles rather than hygiene practice. Several keywords are unambiguous false positives ('Notion' matching 'notion of', 'Sentry' matching Tesla 'sentry mode', 'instrumentation' matching industrial PLC instrumentation, 'playbooks' matching Ansible playbooks, 'protobuf' as a wire format rather than API doc, 'ability to write' matching 'ability to write SQL/code').

_14 drops · 16 guards · 13 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Cross-functional communication & coordination` — 36.2%
- `Observability & telemetry stack` — 29.6%
- `Data-pipeline & data-integration hygiene` — 20.7%
- `Technical documentation authoring & maintenance` — 12.1%
- `Technical writing craft` — 9.8%
- `Product & business literacy` — 9.4%
- `Runbooks, playbooks, & operational docs` — 6.6%
- `System understanding & internal knowledge sharing` — 3.7%
- `Architecture decision records & RFCs` — 3.5%
- `Service-level reliability targets` — 2.4%
- `API & interface documentation` — 1.8%

**Top guards** (false-positive risks worth fixing):
- `monitoring` (Observability & telemetry stack): Bare 'monitoring' co-occurs with admin/maintenance senses: 'Maintenance and Support activities monitoring, reporting', 'Monitoring and improving front-end performance', 'monitoring & alerting' as IT-ops boilerplate. Drives 3107 hits with high noise. → _guard_: Require co-occurrence within a small window (~6 tokens) with one of: logging|alerting|observability|metrics|telemetry|dashboards|traces|logs|prometheus|datadog|grafana|splunk|application performance — i.e., only count 'monitoring' when adjacent to a clear observability collocate.
- `metrics` (Observability & telemetry stack): Catches 'business metrics', 'quality metrics', 'reliability and performance metrics', 'KPIs/metrics' that belong to product-literacy or generic-quality senses, not observability stack. → _guard_: Exclude when preceded by business|product|quality|engineering|customer|user|adoption|hiring|HR|recruiting|growth|financial; or require co-occurrence with logs|traces|dashboards|prometheus|grafana|datadog|telemetry|instrumentation within ~10 tokens.
- `logs` (Observability & telemetry stack): Most snippets are clean ('metrics, logs and traces'), but bare 'logs' can match 'audit logs', 'application logs' debugging, or 'CICS dumps and traces'. → _guard_: Require co-occurrence with metrics|traces|observability|aggregation|centralized|structured|telemetry|elastic|kibana|splunk|fluentd within ~8 tokens; otherwise downweight.
- `traces` (Observability & telemetry stack): Matches 'sniffer traces' (network packets), 'CICS dumps and traces' (mainframe debugging), unrelated to distributed tracing. → _guard_: Require co-occurrence with metrics|logs|distributed|opentelemetry|jaeger|honeycomb|datadog|tempo within ~8 tokens.
- `alerts` (Observability & telemetry stack): Matches Workday 'Alerts and Notifications framework', Salesforce 'alerts', security 'custom alerts' for SOC playbooks — not necessarily observability alerting. → _guard_: Require co-occurrence with monitoring|alerting|metrics|threshold|pagerduty|opsgenie|on-call|incident within ~8 tokens; exclude when adjacent to 'notifications' as part of Workday/Salesforce framework names.

**Top suggested additions** (grounded in corpus snippets):
- `SOPs` → Runbooks, playbooks, & operational docs
- `standard operating procedures` → Runbooks, playbooks, & operational docs
- `TSGs` → Runbooks, playbooks, & operational docs
- `AppDynamics` → Observability & telemetry stack
- `CloudWatch` → Observability & telemetry stack

**Concept-level recommendations:**
- `Architecture decision records & RFCs`: The concept is currently dominated by 'technical specifications' (419), 'design artifacts' (83), and 'architecture reviews' (71) — terms whose snippets show requirements-to-spec translation, generic SDLC deliverables, and code-quality reviews. The deliberative-artifact sense (ADR/RFC) is genuinely rare in postings (ADRs: 10, RFCs: 10, architecture decision records: 5). The hit_rate of 3.5% is therefore inflated. → _Either (a) tighten the concept to true ADR/RFC artifacts and accept a real concept_hit_rate near 1%, dropping 'technical specification(s)/proposal(s)/specs' as drops above; or (b) rename the concept to 'Design proposals & deliberative artifacts' and explicitly include design docs, design reviews, and engineering proposals as in-scope while still removing the requirements-translation snippets via guards._
- `Data-pipeline & data-integration hygiene`: The concept mixes hygiene/governance language ('data quality', 'data lineage', 'data contracts', 'data dictionaries', 'data stewardship', 'data observability') with raw data-engineering tool nouns ('snowflake' 794, 'databricks' 657, 'redshift' 399, 'bigquery' 267, 'airflow' 525, 'dbt' 224). The latter primarily mark data-engineering job postings rather than hygiene-as-a-skill, and they collectively dominate the concept's hit_rate. → _Split the concept into (a) 'Data hygiene & governance practices' (lineage, contracts, validation, dictionaries, stewardship, quality checks, observability, catalog, mesh) and (b) 'Data platform tools' (snowflake, databricks, redshift, bigquery, airflow, dbt, dagster, prefect, lakehouse). The hygiene concept then yields a clean rate-of-practice signal; the tools concept can be kept as proxy if needed but should not be conflated with hygiene._
- `Cross-functional communication & coordination`: Two of the largest keywords ('stakeholders' 3994, 'cross-functional' 3453) are filler boilerplate. Many postings include them without any specific coordination practice, inflating concept_hit_rate to 36% — likely the highest-noise concept in the topic. → _Either (a) add proximity guards (require stakeholders/cross-functional to be within ~6 tokens of manage|align|partner|translate|non-technical|communicate to count); or (b) downgrade 'stakeholders' and 'cross-functional' to count-with-half-weight relative to more specific phrases ('non-technical stakeholders', 'translate technical concepts', 'partner with product/design', 'cross-functional partners') so the concept is anchored on coordination acts rather than mere mention._
- `Technical writing craft`: 'ability to write' (362 hits) and 'written communication' (1123 hits) drive most of this concept; 'ability to write' almost always refers to writing code/SQL, not prose. After dropping 'ability to write', 'written communication' alone may be too generic to distinguish 'writing craft' from generic communication. → _Drop 'ability to write' (already in drops). Either rename concept to 'Written communication' and accept the broader scope, or guard 'written communication' to require collocates (clear|excellent|strong|technical|design|documents) to keep the craft framing. Without this, the concept and 'Cross-functional communication' overlap heavily on 'written communication'._
- `API & interface documentation`: Concept is partially hijacked by general API/serialization tech rather than documentation practice. 'protobuf', 'GraphQL schema', 'JSON schema' are about interface implementation, not docs. After drops, the concept hit_rate (1.8%) will fall further. → _Tighten concept to documentation artifacts only (Swagger, OpenAPI, API docs/specs/specifications/references/contracts, developer portals/Backstage, Postman collections, interface documentation). Accept lower hit_rate as a true reflection of how rarely postings call out API-doc authoring as a skill._

## Cross-list reconciliation

`collisions.json` lists every keyword appearing in ≥2 topics. The patterns below cluster the most common collisions and propose a canonical home for each.

**Top topic-pair collision counts:**

| Topic A | Topic B | # collisions |
|---|---|---:|
| context_infrastructure | verification | 40 |
| context_infrastructure | orchestration | 11 |
| context_infrastructure | process_scaffolding | 11 |
| orchestration | process_scaffolding | 9 |
| mentorship | verification | 9 |
| performance | verification | 8 |
| context_infrastructure | mentorship | 8 |
| orchestration | verification | 7 |
| process_scaffolding | verification | 7 |
| mentorship | people_management | 4 |

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
