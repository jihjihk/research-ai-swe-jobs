# Vocab Lists — Calibration Review

_Generated 2026-05-06T01:18:01+00:00._


Calibration sample: **19,433 SWE postings** stratified across `kaggle_arshkon/2024-04`, `kaggle_asaniczka/2024-01`, `scraped/2026-04`, matched against `description_core_llm`.


## Executive summary

- Input vocabulary: **2,330 keywords** across **88 core concepts** in **8 topics**.
- Recommended **drops:** 138 keywords (~6% of input — mostly zero-hit entries).
- Recommended **guards:** 124 keywords with high false-positive risk visible in example matches.
- Recommended **additions:** 101 new keywords surfaced from corpus inspection.
- **Cross-list collisions:** 150 keywords appear in 2+ topics; 6 reconciliation rules proposed below.

### Per-topic state

| Topic | Concepts | Keywords | Zero-hit | Drop | Guard | Add | Concept hit-rate range |
|---|---:|---:|---:|---:|---:|---:|---|
| **people_management** | 10 | 125 | 2 | 22 | 18 | 11 | 0.0% — 6.0% |
| **orchestration** | 10 | 204 | 1 | 27 | 16 | 18 | 0.0% — 53.8% |
| **verification** | 9 | 358 | 1 | 8 | 19 | 9 | 4.3% — 40.7% |
| **mentorship** | 11 | 236 | 2 | 24 | 21 | 9 | 0.1% — 21.8% |
| **performance** | 14 | 469 | 116 | 26 | 14 | 12 | 0.3% — 28.5% |
| **process_scaffolding** | 11 | 346 | 1 | 13 | 18 | 10 | 9.1% — 52.4% |
| **legacy_stack** | 12 | 220 | 4 | 4 | 6 | 18 | 0.2% — 3.4% |
| **context_infrastructure** | 11 | 372 | 5 | 14 | 12 | 14 | 2.1% — 36.2% |

## Per-topic findings

### People-management markers (`people_management`)

The vocabulary conflates three distinct constructs: formal people management (direct reports, perf reviews, hiring authority), technical/lead-IC leadership (tech lead, lead software engineer, technical leads), and IC-level mentorship (mentor junior, coaching, career development). High-volume keywords like 'mentor junior' (551), 'coaching' (333), 'lead software engineer' (294), 'team leadership' (123), and 'lead a team' (216) are dominated by IC and senior-IC postings, not people-management roles, which inflates the topic by an order of magnitude beyond the genuine people-mgmt signal (~5-10% of postings, judging by 'people management', 'engineering manager', 'manage a team of', and 'direct reports'). Several role-title keywords (engineering manager, head of engineering, line manager) frequently capture the candidate's manager rather than the candidate's role, and pay-disclosure boilerplate ('compensation decisions') and product-talk ('headcount leverage', 'scaling without scaling headcount') generate false positives. The 'Coaching, mentoring, and career development' concept is the largest source of measurement error and should be split or moved out of people-management entirely. As currently composed, this topic over-counts people management by 3-5x and is not trustworthy for the paper without restructuring; the strict-core subset (direct reports, people management, engineering manager, manage a team of, lead and manage, performance evaluations, perf appraisals, weekly 1:1, hiring decisions) is defensible.

_22 drops · 18 guards · 11 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Coaching, mentoring, and career development` — 6.0%
- `Team leadership` — 5.8%
- `People management role` — 2.3%
- `Team-size signals` — 1.3%
- `Supervisory verbs` — 0.8%
- `Headcount and hiring authority` — 0.5%
- `Direct reports` — 0.4%
- `Performance management` — 0.2%
- `1:1 meetings` — 0.1%
- `Termination, hiring/firing authority` — 0.0%

**Top guards** (false-positive risks worth fixing):
- `direct reports` (Direct reports): Negated forms: 'no direct reports', 'will not have direct reports', 'no direct reports required', 'no direct reports to this role'. → _guard_: Exclude any match preceded within 5 tokens by 'no', 'without', 'not have', 'won't have', 'will not have'. Also drop 'direct reports' when followed within 3 tokens by '?' or 'No'.
- `engineering manager` (People management role): Frequently refers to a separate person the candidate works with ('Work with the Engineering Manager', 'report to the Engineering Manager') or appears as a degree/role-title list option, not the candidate's role. → _guard_: Restrict to first-person/role-defining contexts: 'as an engineering manager', 'engineering manager role', 'you will be an engineering manager', or job-title position ('Software Engineering Manager' in title field). Exclude 'work with', 'report to', 'partner with' preceding within 5 tokens.
- `team leadership` (Team leadership): Listed as a generic soft skill ('Project leadership and management skills. Team leadership skills.') or technical-track competency, not formal people management. → _guard_: Require co-occurrence with people-mgmt evidence in the same posting (manage/lead + team of N, direct reports, performance reviews) to count as people-mgmt rather than soft-skill listing.
- `team lead` (Team leadership): Many examples describe team leads as collaborators ('work with team lead', 'directed by team lead') rather than the candidate's authority; SCRUM master / project-management contexts are common. → _guard_: Drop matches preceded by 'with', 'by', 'from', 'to' within 3 tokens (collaborator framing) and require role-defining context ('as a team lead', 'team lead responsibilities').
- `lead a team` (Team leadership): Captures 'ability to technically lead a team' (technical guidance, IC-track) and generic competency listings. → _guard_: Drop when preceded by 'technically' within 2 tokens or by 'ability to' alone without further people-mgmt anchors. Keep when followed by 'of N engineers/developers' (team-size signal).

**Top suggested additions** (grounded in corpus snippets):
- `Senior Manager` → People management role
- `Engineering Manager` → People management role
- `career growth` → Coaching, mentoring, and career development
- `set goals` → Performance management
- `set performance goals` → Performance management

**Concept-level recommendations:**
- `Coaching, mentoring, and career development`: Vastly over-broad: 'mentor junior' (551 hits), 'coaching' (333), 'mentor engineers' (148), 'mentoring engineers' (98), 'career development' (64) fire on virtually every senior-IC posting. These signals overwhelmingly capture senior-IC mentorship, not formal people management. As implemented, this concept alone drives the topic's 5.97% hit rate, far above any genuine people-mgmt prevalence. → _Either (a) remove this concept from people-management entirely and treat IC mentorship as a separate construct, or (b) narrow it to phrases requiring explicit subordinate framing: 'career growth of your team', 'develop your team', 'career progression plans', 'individual development plan' AND require co-occurrence with at least one strict-core people-mgmt anchor (direct reports / engineering manager / manage a team of / 1:1) in the same posting. Drop the umbrella keywords ('coaching', 'mentor junior', 'mentor engineers', 'mentoring engineers', 'career development') from this concept regardless._
- `Team leadership`: Conflates three distinct things: (1) formal team leads with people-mgmt authority, (2) IC-track senior titles ('Lead Software Engineer', 'Lead Developer', 'Technical Lead'), and (3) collaborator references ('work with technical leads'). The 0.0575 hit rate is dominated by (2) and (3). → _Split into 'Tech-lead-manager hybrid roles' (player-coach, Tech-Lead-Manager, hands-on manager — keep) versus 'IC-track lead titles' (lead software engineer, lead developer, technical leads, engineering leads — drop or move out of people-mgmt). Drop collaborator-context matches. Retain 'lead a team of N' under team-size signals where it has clearest scope._
- `People management role`: Title strings like 'engineering manager', 'line manager', 'people manager' frequently match the candidate's manager or organizational labels rather than the candidate's role. → _Add a positional/role-defining requirement: matches in the title field count strongly; in the body they should be preceded by 'as a/an', 'you will be a/an', 'this is a/an', or 'role of a/an' within 5 tokens. Drop matches preceded by 'work with', 'report to', 'partner with', 'collaborate with'._
- `1:1 meetings`: 'one-on-one' and 'one on one' frequently fire on communication-style phrasing ('communicate in one-on-one settings') and informal mentorship, not regular manager-report cadence. Concept is correctly defined but undercut by overbroad string matches. → _Drop 'one on one' (1 hit, FP). Require 'one-on-one' to be in a meeting/cadence context: preceded by 'weekly', 'regular', 'biweekly', 'monthly', or 'hold' within 5 tokens, or followed by 'meeting', 'meetings', 'with direct reports', 'with reports'. Keep '1:1s', 'weekly 1:1', 'regular one-on-ones' as-is — they are unambiguous._
- `Headcount and hiring authority`: Many growth/scaling phrases ('grow the team', 'expand the team', 'scale the team', 'team growth', 'building a team') ambiguously denote either headcount expansion (people-mgmt) or skill/output growth (any senior IC). Hiring-decisions boilerplate also pollutes. → _Narrow to phrases requiring explicit hiring/recruiting context: keep 'hire and onboard', 'hire and develop', 'recruiting and interviewing', 'interview loops', 'lead hiring', 'leading hiring', 'drive hiring', 'team scaling' (new). Demote or drop the ambiguous growth verbs unless paired with 'recruit', 'hire', 'onboard', 'interview' in same sentence. Apply boilerplate guard on 'hiring decisions' and 'compensation decisions'._

### Orchestration (`orchestration`)

The classical-orchestration concepts (Specs, Architecture, Workflow/Pipeline) are dominated by structural false positives because their highest-firing keywords (specifications, architecture, architectural, pipeline, pipelines, orchestration, workflow, workflows, design patterns) match commodity SDLC and infrastructure language (FDA/ELF specs, Kubernetes container orchestration, CI/CD pipelines, GoF design patterns, ServiceNow workflow forms) rather than the senior 'orchestrator' archetype the topic targets. The classical-planning concept also leans on generic soft-skill phrases (ambiguity, break down, scoping, roadmap) that fire on every SWE posting. By contrast, the AI-side concepts (Context Engineering, Agent Harnesses, Repo Instructions, Memory, Multi-Agent Coordination, Eval/Steering) are narrowly scoped to scraped 2026 postings and most keywords look clean, with only a few cross-domain collisions (multi-agent in robotics, ACP as a SAFe certification, tool use in flight-software). To make the topic discriminative, drop or guard the broad classical terms, narrow architecture/spec/workflow keywords to authoring-style verbs and named artifacts (RFCs, ADRs, PRDs, design document, write specifications, spec-driven), and add observed AI-orchestration phrases (skills, A2A, function calling, context injection, agentic loops, agentic patterns, tool orchestration, agent lifecycle).

_27 drops · 16 guards · 18 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Software Architecture and System Design` — 53.8%
- `Workflow Design and Pipeline Orchestration` — 39.9%
- `Specifications and Requirements Authoring` — 17.0%
- `Task Decomposition and Planning` — 12.4%
- `Agent Harnesses and Coding Agents` — 5.6%
- `Context Engineering and Prompt/Spec Authorship for Agents` — 2.6%
- `Multi-Agent Coordination` — 1.1%
- `AI Tool and Agent Evaluation/Steering` — 0.8%
- `Agent Memory and State Management` — 0.1%
- `Repository Instructions and Agent Configuration Files` — 0.0%

**Top guards** (false-positive risks worth fixing):
- `architecture` (Software Architecture and System Design): Matches 'microservices architecture', 'event-driven architecture', containerization 'orchestration architecture' — i.e., naming an architectural style is not the same as authoring/leading architecture. → _guard_: Require co-occurrence with authoring/leadership verbs (design, define, lead, own, drive, author, establish) within a small window, or restrict to phrases like 'architecture decisions', 'architecture reviews', 'architecture documentation', 'reference architecture'.
- `architectural` (Software Architecture and System Design): 'architectural principles' / 'architectural patterns' as part of generic JD boilerplate, or 'architectural skills' as a soft-skill bullet. → _guard_: Restrict to 'architectural decisions', 'architectural review(s)', 'architectural standards', 'architectural direction'.
- `architect` (Software Architecture and System Design): Frequently the job title 'Java architect / Solutions Architect' — title-card not skill description. → _guard_: Exclude when match is in title field; require a verb usage ('architect and build', 'architect solutions', 'architect for scalability') in the body.
- `architectures` (Software Architecture and System Design): Plural enumeration of styles ('microservices architectures', 'enterprise architectures', 'machine learning architectures'). → _guard_: Require co-occurrence with design/own/lead verbs, or with 'reference architectures', 'multiple architectures'.
- `architecting` (Software Architecture and System Design): Usually OK ('architecting Azure cloud solutions') but can blur with implementation; mostly fine. Consider tightening if recall is too broad. → _guard_: Optionally guard with object phrases like 'architecting solutions/systems/platforms' to exclude rare 'architecting code for memory'.

**Top suggested additions** (grounded in corpus snippets):
- `skills` → Repository Instructions and Agent Configuration Files
- `system prompt` → Context Engineering and Prompt/Spec Authorship for Agents
- `context injection` → Context Engineering and Prompt/Spec Authorship for Agents
- `prompt orchestration` → Context Engineering and Prompt/Spec Authorship for Agents
- `function calling` → Agent Harnesses and Coding Agents

**Concept-level recommendations:**
- `Specifications and Requirements Authoring`: After dropping the generic '*requirements' nouns, the remaining keywords lean on the word 'spec' which in SWE postings overwhelmingly means hardware/protocol/FDA specification rather than authored system specs. Concept conflates 'reads/translates requirements' (generic SDLC) with 'authors specs/RFCs/PRDs' (the orchestrator archetype the topic intends). → _Restrict the concept to authorship verbs + named artifacts: 'write specifications', 'author RFCs', 'write design document(s)', 'PRDs', 'spec-driven', 'architecture decision records', 'ADRs', 'design spec(ification)', plus phrases like 'requirements document authoring', 'functional specification authoring'. Drop bare requirement-reading nouns. Consider renaming the concept 'Spec/RFC Authorship' to make the intent explicit._
- `Software Architecture and System Design`: The concept currently fires on 53.8% of postings — almost any senior SWE JD that mentions 'architecture' or 'design patterns' lights up. This dilutes the orchestrator signal because naming an architectural style (microservices, event-driven) is being conflated with authoring/leading architecture. → _Rebuild the concept around authoring verbs and named artifacts: 'design and own X', 'lead architecture', 'architectural decisions/reviews/standards', 'reference architecture', 'ADRs', 'HLD/LLD', 'design document', 'system design', 'distributed systems design'. Drop bare style nouns ('architecture', 'architectures', 'architectural', 'architectural patterns', 'design patterns', 'API design', 'service design')._
- `Task Decomposition and Planning`: Half of the concept's hits come from generic soft-skill phrasing ('comfort with ambiguity', 'break down complex problems', 'roadmap', 'project planning') that fires on every senior JD. The concept loses discriminative power. → _Drop generic phrases (ambiguity, ambiguous problems, break down, breaking down, scoping, roadmap(s), plan and execute, execution plan/planning). Keep only specific decomposition vocabulary: decompose, decomposition, task decomposition, problem decomposition, work breakdown, technical roadmap, technical planning, project planning. Consider adding 'sequencing work' and 'epics and stories' (observed in Java Architect: 'creating epics and stories for developers') as more concrete planning artifacts._
- `Workflow Design and Pipeline Orchestration`: Top keywords ('workflow', 'workflows', 'pipeline', 'pipelines', 'orchestration', 'orchestrators') fire on commodity DevOps/CI-CD/Kubernetes language rather than the senior workflow-design archetype. CI/CD pipelines in particular dominate the 4549 'pipelines' hits. → _Either (a) split the concept into 'Data/ML pipeline orchestration' (Airflow, Prefect, Dagster, Argo Workflows, data pipelines, ML pipelines, DAG/DAGs, workflow orchestration, pipeline orchestration) versus 'Workflow design as authorship' (workflow design, process orchestration, workflow engine), or (b) keep the concept but require pairing the broad nouns with authoring verbs (design/build/author/orchestrate) or named tools. As-is, 'pipeline'/'pipelines'/'workflow'/'workflows' alone should not count._
- `Multi-Agent Coordination`: Robotics/defense postings (UAVs, mission planning, simulation) match 'multi-agent', 'multi-agent systems', and 'multi-agent coordination' even though the topic targets LLM-agent coordination. ACP keyword is fully captured by SAFe/PMI-ACP certification, contributing zero true positives. → _Add an LLM/agentic-AI co-occurrence guard for the 'multi-agent*' keyword family. Drop ACP. Add 'A2A' / 'Agent-to-Agent' / 'Agent2Agent' as the AI-agent communication protocol. Consider adding observed concrete patterns: 'supervisor pattern(s)', 'planner pattern', 'role-based agents'._

### Verification (`verification`)

The verification vocabulary is broad and mostly well-grounded for core testing/QA/CI/compliance/observability lexicons, but several high-volume keywords are heavily contaminated by adjacent meanings: bare 'pipeline'/'pipelines', 'metrics'/'metric', 'dashboard'/'dashboards', 'alert'/'alerts', 'monitors'/'monitored', 'tested', 'artifact'/'artifacts', and 'tracing'/'traces' all match generic data-engineering, BI, or risk-management language as often as they match observability/CI/QA. The Reproducibility-and-artifact-proof concept is the weakest fit: most of its high-hit terms ('log', 'logs', 'logging', 'tracing', 'traces') are really observability vocabulary that already lives under Post-deployment observability, while 'artifact'/'artifacts' overwhelmingly denote generic engineering deliverables (test artifacts, code artifacts, design artifacts) rather than evidence-of-correctness. The Evaluations-and-AI-output-verification concept is diluted by polysemous terms ('benchmark', 'benchmarking', 'guardrails', 'gold standard', 'red team') that fire mostly on performance benchmarking, security guardrails, idiomatic usage, or offensive-security red teams. Recommended fixes are mostly guards (require co-occurrence with disambiguating tokens) rather than drops, plus shifting log/trace terms out of artifact-proof and into observability or dropping the artifact-proof concept altogether.

_8 drops · 19 guards · 9 adds · 4 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `CI/CD and quality gates` — 40.7%
- `QA and validation` — 29.2%
- `Automated testing` — 24.2%
- `Post-deployment observability` — 21.9%
- `Compliance and security verification` — 16.8%
- `Code review` — 16.0%
- `Reproducibility and artifact proof` — 10.1%
- `Static analysis and code quality tooling` — 7.0%
- `Evaluations and AI output verification` — 4.3%

**Top guards** (false-positive risks worth fixing):
- `pipeline` (CI/CD and quality gates): Matches 'data pipeline', 'ML pipeline', 'ETL pipeline', 'machine learning pipeline', 'analytics pipeline'. → _guard_: Require token within ~5 tokens of CI|CD|build|deploy|release|Jenkins|GitLab|Azure DevOps|GitHub Actions|continuous; or require pluralization 'pipelines' co-occurring with one of these tokens. Drop hits where preceded by 'data'|'ML'|'ETL'|'analytics'|'machine learning'|'inference'|'training'.
- `pipelines` (CI/CD and quality gates): Same as 'pipeline': data/ML/ETL pipelines dominate counts as much as CI/CD pipelines do. → _guard_: Same disambiguation: require nearby CI|CD|build|deploy|release|Jenkins|GitLab CI|Azure DevOps|GitHub Actions|continuous integration|continuous delivery.
- `metrics` (Post-deployment observability): Matches 'business metrics', 'key business metrics', 'product metrics', 'quality metrics' and other non-observability uses. → _guard_: Require co-occurrence (within ~10 tokens) of one of: dashboards|alerts|alerting|logs|tracing|monitor|monitoring|observability|SLO|SLI|Datadog|Grafana|Prometheus|CloudWatch|New Relic|Splunk.
- `dashboards` (Post-deployment observability): Heavily fires on BI/analytics dashboards (Tableau, Power BI, reporting dashboards) and QA dashboards rather than ops dashboards. → _guard_: Require co-occurrence with monitor|monitoring|observability|alerts|metrics|Grafana|Datadog|Kibana|CloudWatch|Splunk; exclude when preceded by 'Tableau'|'Power BI'|'reporting'|'BI'|'business intelligence'|'analytics'.
- `dashboard` (Post-deployment observability): Same as 'dashboards' and worse — frequent BI/Tableau/Power BI uses. → _guard_: Same observability co-occurrence guard as 'dashboards'.

**Top suggested additions** (grounded in corpus snippets):
- `ELK` → Post-deployment observability
- `Kibana` → Post-deployment observability
- `CloudWatch` → Post-deployment observability
- `X-Ray` → Post-deployment observability
- `shift-left` → CI/CD and quality gates

**Concept-level recommendations:**
- `Reproducibility and artifact proof`: Most of this concept's high-volume keywords (log, logs, logging, traces, tracing) are observability terms that already belong under Post-deployment observability; the genuinely 'proof' keywords (audit trail/trails, audit logs, formal verification, reproducible build/builds, reproducible research, reproduce/reproduces/reproducing) have very low hit counts. As written, the concept's 1959 hits are dominated by observability false positives, not by proof/evidence semantics. → _Either (a) merge log/logs/logging/traces/tracing into Post-deployment observability and shrink this concept to genuine evidentiary terms (audit trail*, audit log*, screenshot, recording, replay, reproducible*, formal verification, evidence) — accepting it will be a small concept, or (b) drop the concept entirely since the artifact-proof signal is too weak to support a top-level construct in SWE postings._
- `Evaluations and AI output verification`: Concept is diluted by polysemous terms (benchmark/benchmarking, guardrails, red team, gold standard) that overwhelmingly fire on non-AI senses; the AI-specific keywords (LLM evals, model evaluation, eval-driven, hallucinations, LangSmith, RAGAS, etc.) are the real signal but are individually low-hit. → _Tighten by requiring AI/LLM/model/prompt/agent/GenAI co-occurrence for the polysemous terms (benchmark*, guardrails, red team, gold standard, ground truth), or restrict the concept's keyword set to AI-marked phrases (LLM evals, model evaluation, eval harness, AI evaluation, LLM-as-a-judge, plus AI-eval tools). This makes the concept more precise at some cost in recall._
- `Static analysis and code quality tooling`: 'code quality' (895 hits) overwhelmingly dominates this concept's volume but is a generic phrase used in code-review/QA contexts more than in static-analysis tooling contexts; the actual static-analysis tools (linters, SAST, SonarQube, etc.) are individually small. The aggregate hit rate is therefore misleading. → _Either move 'code quality' to Code review or QA-and-validation, or split this concept into two: (a) static analysis tooling (linters/SAST/type-checkers/specific tools) and (b) generic code-quality language. Reporting hit rates separately would be more informative._
- `Post-deployment observability`: The high-volume vocabulary ('metrics', 'dashboards', 'alerts', 'monitors') is genuinely observability vocabulary in many postings but bleeds into BI/analytics dashboards, business-metric language, and risk-management 'monitored' usage. Without guards, the concept overstates observability prevalence. → _Apply the per-keyword guards listed above (require co-occurrence with at least one observability anchor token: monitor*|alert*|on-call|incident|SLO|SLI|Datadog|Grafana|Prometheus|CloudWatch|Kibana|Splunk|New Relic|production). Consider also adding ELK/Kibana/CloudWatch/X-Ray as anchor tokens._

### Mentorship markers (`mentorship`)

The 'Direct mentorship language', 'Onboarding and ramp-up', and 'Pair / mob / ensemble programming' concepts are mostly clean and well-targeted. The most damaging false-positive sources are 'Coaching and developing engineers' (where 'coach', 'coaching', and 'develop talent' fire on people-management/HR contexts and on coaching customers/end-users), and 'Architecture and design guidance' (where 'technical direction', 'drive best practices', 'establish best practices', 'define standards', 'drive architecture', 'shape the architecture' frequently describe solo senior-IC ownership rather than guidance directed at other engineers, blurring this concept into Verification/architecture). 'Explicit learning culture' is severely overbroad: 'curiosity', 'continuous learning', 'continuously learn', 'professional development', 'professional growth', 'career growth', 'career development', 'personal development', 'personal growth', 'skill development', 'self-improvement', 'always learning', 'love to learn', 'lifelong learner', 'develop new skills', 'growth-oriented', 'engineering culture' nearly all fire on the candidate's own self-improvement disposition or on generic perks/benefits language, not on a team-level learning culture. 'Communication and feedback skills' has 'feedback loop' (almost entirely ML/CI/customer feedback-loop FPs) and 'force multiplier' (mostly product/marketing copy). 'Code review as teaching' concept also leaks into Verification through 'review code', 'peer review', 'peer reviews', 'reviewing code', 'pr reviews' which fire on plain correctness-gate review with no teaching context.

_24 drops · 21 guards · 9 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Direct mentorship language` — 21.8%
- `Guidance and influence` — 13.1%
- `Explicit learning culture` — 7.1%
- `Code review as teaching` — 5.1%
- `Teaching and knowledge transfer` — 4.8%
- `Coaching and developing engineers` — 4.7%
- `Architecture and design guidance` — 4.5%
- `Onboarding and ramp-up` — 2.8%
- `Pair / mob / ensemble programming` — 1.0%
- `Communication and feedback skills` — 0.8%
- `Debugging and code-reading as a craft` — 0.1%

**Top guards** (false-positive risks worth fixing):
- `coach` (Coaching and developing engineers): Fires on 'coach business stakeholders / end users / non-technical clients' and 'AI Coaching platform' product copy. → _guard_: Require nearby engineer-target token (engineer/developer/junior/team member/peer/staff/IC) within +/-8 tokens; exclude when followed by 'users', 'clients', 'stakeholders', 'customers', or when 'AI coaching' / 'coaching platform' product context.
- `coaching` (Coaching and developing engineers): Fires on 'AI Coaching experiences for users', QA-team coaching of testers, and HR-style 'performance coaching' tied to manager duties. → _guard_: Require co-occurrence with engineer/developer/team-member target; exclude when adjacent to 'users', 'clients', 'AI coaching', 'salary reviews', or 'performance reviews' (those are People-management).
- `coaches` (Coaching and developing engineers): Fires on 'Coaches technology communities at Discover' (broad community evangelism) and on people-manager descriptions. → _guard_: Require engineer-target proximity; drop when paired with 'salary reviews', 'performance reviews', 'communities', or 'stakeholders'.
- `develop talent` (Coaching and developing engineers): Fires on 'develop talent pipelines and succession plans' (HR/recruiting) and software-manager job descriptions. → _guard_: Drop when 'talent pipeline', 'succession plan', 'hiring', or manager-role keywords appear within +/-12 tokens; keep only when an engineer/developer target is named.
- `raise the bar` (Coaching and developing engineers): Fires on 'raise the bar on polish', 'raise the bar for quality / code release' which is quality discipline (Verification), not engineer growth. → _guard_: Keep only when followed by 'for the team' or 'for those around', or co-occurs with engineer/team/colleague target; otherwise route to Verification or drop.

**Top suggested additions** (grounded in corpus snippets):
- `code review that teaches` → Code review as teaching
- `teach by example` → Teaching and knowledge transfer
- `career development discussions` → Coaching and developing engineers
- `pairing and design reviews` → Pair / mob / ensemble programming
- `hands-on coaching` → Coaching and developing engineers

**Concept-level recommendations:**
- `Architecture and design guidance`: The concept is supposed to capture senior-IC architecture decisions made FOR/WITH other engineers (mentorship-flavored), but several high-hit keywords ('technical direction', 'drive architecture', 'shape the architecture', 'establish best practices', 'drive best practices', 'define standards', 'establish patterns') fire heavily on solo-IC architectural ownership with no audience-of-engineers signal. As written, it largely overlaps with the Architecture/Verification topic rather than Mentorship. → _Tighten the concept to require an audience-of-engineers cue: keep a hit only when the architecture/standards verb co-occurs with 'team', 'engineers', 'mentor', 'guide', 'across the team', 'pairing', 'design reviews', or similar within a small window. Move bare solo-architecture variants to the Architecture topic. Consider splitting this concept into 'Architecture guidance to other engineers' (kept here) vs 'Architecture decision authority' (moved out)._
- `Explicit learning culture`: The concept conflates three distinct things: (a) candidate personal traits ('curiosity', 'intellectually curious', 'lifelong learner', 'self-improvement', 'love to learn'), (b) HR/benefits language ('professional development', 'career development', 'career growth', 'learning opportunities'), and (c) genuine team-level learning culture ('culture of learning', 'learning culture', 'learn from each other', 'blameless postmortem'). Bucket (b) is the single largest source of false positives; bucket (a) is a mismatch with the concept's stated 'team/company/role' framing. → _Restrict to bucket (c) only: drop the perks/benefits keywords and the candidate-trait keywords. Optionally split out a separate 'Personal learning disposition' concept if that signal is wanted, but keep it explicitly distinct from team-level learning culture. Require team-scope qualifiers ('culture of', 'team that', 'environment of', 'company that') for the borderline phrases, and keep blameless postmortem / learn-from-failures / learn-from-each-other as the strongest team-level markers._
- `Code review as teaching`: The bare review-process keywords ('review code', 'reviewing code', 'peer review', 'peer reviews', 'pr reviews', 'code-review', 'design discussions', 'provide feedback') fire heavily on plain Verification-style review with no teaching context. The concept's note about overlap with Verification correctly anticipates this, but the operationalization does not enforce the 'paired with feedback/learning/growth language' criterion. → _Make the teaching cue mandatory rather than implied: require any of these review keywords to co-occur with a teaching-cue token ('mentor', 'teach', 'grow', 'learn', 'junior', 'feedback', 'uplift', 'develop') within a small window. Otherwise route to Verification. Promote the strong-teaching variants ('thoughtful code review', 'review that teaches', 'deep-dive code reviews') as standalone signals not requiring the cue._
- `Communication and feedback skills`: The concept aims at engineer-to-engineer feedback skill, but several keywords ('feedback loop', 'force multiplier', 'empathetic', 'direct feedback') do not encode that target. This concept already has the lowest hit rate (0.008), so pruning will mostly remove FPs rather than signal. → _Drop 'feedback loop' (almost entirely ML/CI/customer feedback), 'force multiplier' (product/leverage copy), 'empathetic' (personality trait), 'direct feedback' (stakeholder feedback). Keep 'constructive feedback', 'giving and receiving feedback', 'open to feedback', 'candid feedback', 'make others better', 'make those around you better', 'multiply other engineers', 'high-feedback culture'. Consider absorbing this concept into 'Code review as teaching' since the engineer-feedback signal heavily overlaps._
- `Debugging and code-reading as a craft`: Concept has only 28 postings (0.001) and 3 keywords; 'reading code' duplicates a keyword in 'Code review as teaching'; 'walkthroughs' / 'code walkthroughs' fire on architectural/spec walkthroughs in waterfall/SDLC contexts (compliance review), not on a learning-by-reading craft. → _Either retire this concept (folding any kept signal into 'Code review as teaching') or sharpen it: 'reading code' should require co-occurrence with 'learn', 'understand', 'mentor', or 'unfamiliar codebase'. Add stronger debugging-as-craft phrases ('debugging together', 'walk through the code', 'read code with') if found in further sweeps; drop 'walkthroughs' bare (too procedural)._

### Performance & deep technical understanding (`performance`)

The vocabulary captures performance/internals work reasonably well but has substantial false-positive leakage on a handful of high-firing keywords that dominate concept hits. The depth_claim_language concept (28.5% topic-wide) is largely capturing generic ad-copy fluff ('strong understanding of HTML5', 'solid understanding of DevOps') that has nothing to do with low-level performance, distorting the topic. Several keywords match unrelated domains entirely: 'real-time' (1,492 hits, mostly real-time data/streaming/AI), 'instrumentation' (mostly industrial/SCADA/test instrumentation), 'tracing' (mostly requirements-tracing or data tracing), 'profiling' (substantial 'data profiling'/'profiler' as in SQL Profiler), 'BGP' under networking_internals (network-engineer routing, not internals), 'JIT' (mostly Just-In-Time access in IAM contexts), 'capacity planning'/'cost optimization'/'cost efficiency' (FinOps/cloud admin, not perf engineering), and 'horizontal/vertical scaling' (high-level system design). Genuine low-level signal keywords (DPDK, RDMA, eBPF, XDP, SIMD, NUMA, kernel internals, memory layout, lock-free, microarchitecture) fire faithfully but at very low rates, suggesting the internals-perf demand is genuinely small in this sample. 116 of 469 keywords have zero hits, including most theoretical/academic distributed-systems and database-internals terms.

_26 drops · 14 guards · 12 adds · 5 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `depth_claim_language` — 28.5%
- `latency_throughput_scale` — 13.5%
- `performance_optimization_general` — 9.1%
- `distributed_systems_internals` — 7.3%
- `profiling_and_benchmarking` — 6.5%
- `scaling_and_efficiency` — 4.7%
- `concurrency_parallelism` — 4.2%
- `low_level_systems_programming` — 3.9%
- `hardware_aware_programming` — 2.2%
- `os_kernel_internals` — 1.3%
- `database_storage_internals` — 0.9%
- `networking_internals` — 0.7%
- `algorithmic_optimization` — 0.6%
- `compilers_and_runtimes` — 0.3%

**Top guards** (false-positive risks worth fixing):
- `real-time` (latency_throughput_scale): Dominated by 'real-time data acquisition/streaming/processing' (Kafka/streaming) and AI 'real-time insights' rather than RT perf constraints; only a minority refer to hard/soft real-time computing. → _guard_: Require co-occurrence with one of: 'RTOS', 'hard real-time', 'soft real-time', 'real-time systems', 'real-time embedded', 'real-time control', or perf-context tokens (latency, deadline, deterministic). Drop or down-weight the standalone 'real-time' substring.
- `realtime` (latency_throughput_scale): Same pattern: 'Realtime time streaming', 'realtime APIs', 'realtime server web'. → _guard_: Same as above — require RT-systems co-occurrence.
- `profiling` (profiling_and_benchmarking): Frequent 'data profiling' (data quality/EDA), 'deep data profiling', and 'application performance profiling' (legitimate but already covered). → _guard_: Exclude when preceded by 'data ' or 'user '. Keep when adjacent to perf/CPU/memory/system/code or co-occurs with 'bottleneck'/'optimize'.
- `profiler` (profiling_and_benchmarking): 11 hits dominated by 'SQL Profiler' (a SQL Server tool, not a perf profiler in the systems sense) and 'Watson Profiler'. → _guard_: Exclude 'SQL Profiler' explicitly; keep when paired with code/perf/CPU/memory or named perf profilers (gprof, Tracy, PiX, Superluminal).
- `BGP` (networking_internals): BGP fires in network-engineer/routing JDs; the topic claims networking-internals but most BGP mentions are operations/routing-protocol knowledge, not stack-internals. → _guard_: Keep BGP only when co-occurring with stack-internals tokens (DPDK, XDP, eBPF, kernel bypass, dataplane, packet pipeline). Otherwise it pulls in network/SRE roles.

**Top suggested additions** (grounded in corpus snippets):
- `performance engineer` → performance_optimization_general
- `performance critical path` → performance_optimization_general
- `flame graph` → profiling_and_benchmarking
- `Tracy` → profiling_and_benchmarking
- `PiX` → profiling_and_benchmarking

**Concept-level recommendations:**
- `depth_claim_language`: Concept currently fires on 28.5% of postings driven almost entirely by generic boilerplate ('strong understanding of', 'solid understanding of', 'deep understanding of') applied to trivial domain skills (HTML5, OOP, CI/CD). This makes the concept synonymous with 'is a job posting' rather than measuring depth/internals-of-X demand. The intended construct (employer signaling first-principles, expert-level technical depth) is being drowned out. → _Either (a) drop this concept entirely from the perf topic and convert it into a topic-agnostic modifier that boosts a posting's score only when the depth phrase is followed by an internals-domain noun (kernel, runtime, JVM, compiler, distributed systems, GPU, memory, latency, networking stack, etc.), or (b) keep only the high-specificity phrases ('first-principles', 'from first principles', 'under the hood', 'intimately familiar with', 'intimate knowledge of', 'rigorous understanding') and drop the high-volume generic ones._
- `scaling_and_efficiency`: Mixes three different concepts: (1) very-large-scale operational claims ('massive scale', 'scalable systems', 'scale to millions'), (2) perf-resource-efficiency ('memory footprint', 'memory pressure', 'CPU utilization'), and (3) FinOps/cost ('cost optimization', 'cost efficiency', 'capacity planning'). FinOps is a distinct topic from internals/perf and currently dominates concept hits. → _Split: keep 'scaling axis / massive-scale' as one concept; promote 'memory efficiency / footprint / pressure / leak / utilization' into the perf-optimization or low-level-systems concept; remove FinOps terms ('cost optimization', 'cost efficiency', 'capacity planning') from this topic — they belong in a separate cloud/operations topic._
- `latency_throughput_scale`: 'real-time' (1492 hits) and 'high volume'/'high-volume' (315 hits combined) overwhelmingly capture streaming-data and generic-volume framing rather than latency/throughput perf objectives. The concept's hit_rate (13.5%) is largely driven by these noisy strings. → _Drop bare 'real-time'/'realtime' or guard them with an RTOS/RT-systems requirement. Drop 'high volume' (with space) and guard 'high-volume' to require co-occurrence with traffic/transactions. The concept's signal should rely on the metric-named keywords (latency, throughput, p99, QPS, RPS, sub-millisecond) which are far cleaner._
- `profiling_and_benchmarking`: 'instrumentation' (222 hits) and 'tracing' (179 hits) carry strong false-positive rates from unrelated domains (industrial instrumentation, requirements tracing). 'profiling' (345) has moderate data-profiling contamination. → _Drop 'instrumentation' as a bare keyword (replace with 'code instrumentation', 'binary instrumentation', or 'performance instrumentation'). Drop 'tracing' as a bare keyword (replace with 'distributed tracing', 'CPU tracing', 'execution tracing' — note 'distributed tracing' is missing from the list and would be a clean signal). Guard 'profiling' to exclude 'data profiling'._
- `networking_internals`: BGP (40 hits) is the single largest contributor but it is a routing-protocol skill; the concept's actual internals signal lives in DPDK/RDMA/InfiniBand/XDP/eBPF/kernel-bypass — all with very low hit rates (1-10). → _Demote or drop bare 'BGP'; the existing low-volume kernel-bypass keywords are the true signal. Add VPP, SPDK, P4 (see ADDS) to broaden coverage of the dataplane-internals slice that genuinely exists in the data._

### Process-scaffolding markers (`process_scaffolding`)

Several high-hit anchor keywords are massively contaminated by case-insensitive substring matches that have nothing to do with process scaffolding: 'SAFe' (557 hits) almost entirely matches 'safe operation/deployment/maintainable', and 'LeSS' (567 hits) matches the LESS CSS preprocessor and the word 'less' (e.g., 'less experienced staff'); together they are inflating the Kanban/Lean/Waterfall concept by ~1100 spurious postings. The Verification & Validation concept is structurally over-broad: high-hit verbs ('validation', 'validate', 'validating', 'verify') fire on data/model validation in ML and ETL pipelines and on input/data 'verify' tasks, which are not process-level V&V; without a co-occurrence guard this concept will misclassify ML/data postings as process-heavy. Several short-form acronyms pull in unrelated tech stacks: 'CSM' matches ServiceNow Customer Service Management, 'IMS' matches IBM IMS DB mainframe, 'epic' matches Epic Systems healthcare and Unreal Engine 'EPIC platform', and 'RTE' matches AUTOSAR Run-Time Environment far more than Release Train Engineer. The Project-coordination concept's verb stems ('coordinate', 'facilitate', 'liaison') are generic teamwork verbs whose snippets show heavy use as routine SWE collaboration language rather than process-coordination markers, creating a high baseline that hampers signal. Finally, the 'requirements' family is dominated by a single token ('requirements' alone matches 49% of postings) where most usage is just the JD section header 'Requirements:' or generic 'system requirements' — that single keyword is a near-tautology against SWE postings and should be replaced or guarded with phrase-level evidence.

_13 drops · 18 guards · 10 adds · 6 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Requirements engineering` — 52.4%
- `SDLC and process governance` — 32.0%
- `Agile methodology` — 31.2%
- `Scheduling and milestones` — 23.2%
- `Project / program management roles & tooling` — 17.9%
- `Verification & Validation (V&V)` — 16.4%
- `Project coordination` — 15.8%
- `Scrum framework` — 13.4%
- `Specification authoring` — 11.5%
- `Sprints and iterations` — 9.5%
- `Kanban / Lean / Waterfall and other methodologies` — 9.1%

**Top guards** (false-positive risks worth fixing):
- `SAFe` (Kanban / Lean / Waterfall and other methodologies): Case-insensitive matching catches the common adjective 'safe' in phrases like 'safe operation', 'safe deployment practices', 'safe and maintainable factory'. All five visible examples are false positives; true SAFe-as-Scaled-Agile-Framework usage is buried. → _guard_: Require case-sensitive 'SAFe' as a whole token, OR require co-occurrence with 'agile|scaled|framework|train|certification|agilist' within +/-30 chars.
- `LeSS` (Kanban / Lean / Waterfall and other methodologies): Case-insensitive matching catches 'LESS' (CSS preprocessor: 'SASS/LESS', 'CSS Preprocessor (Less or Sass)') and the comparative word 'less' ('less experienced staff', 'guidance to less experienced personnel'). All five visible examples are false positives. → _guard_: Require case-sensitive 'LeSS' (mixed-case as defined by Large-Scale Scrum) AND co-occurrence with 'scrum|scaled|agile|framework' within +/-50 chars; without a guard this keyword is unusable.
- `agile` (Agile methodology): 5,793 hits is plausibly correct for SWE postings, but the bare token also matches 'fast-paced, agile environment' — generic adjectival usage that conflates a culture buzzword with methodology adoption. Many snippets pair 'agile' with no real process content. → _guard_: Keep, but flag 'agile' alone as a weak signal; require co-occurrence with one of {scrum, kanban, sprint, ceremonies, methodology, framework, SAFe, lean} for high-confidence process-scaffolding scoring.
- `lean` (Kanban / Lean / Waterfall and other methodologies): Even after dropping bare 'lean' for the methodology sense, downstream consumers should note 'lean DevOps', 'lean startup environment' are culture, and 'Lean/Six Sigma' is quality methodology not process methodology. → _guard_: Replace bare 'lean' with phrase-level keywords only: 'lean software development', 'lean principles', 'lean startup', 'lean methodology'.
- `stand up` (Scrum framework): Phrase 'stand up' fires on unrelated verb usage: 'stand up data platforms', 'stand up data systems' (provision/deploy). 5/5 visible examples include several false positives of this type. → _guard_: Drop bare 'stand up'; keep only 'daily stand up' / 'stand-up' / 'standup' / 'daily standup'. Or require co-occurrence with 'daily|meeting|scrum|ceremony'.

**Top suggested additions** (grounded in corpus snippets):
- `ADRs` → Specification authoring
- `SRD` → Requirements engineering
- `ASPICE` → Verification & Validation (V&V)
- `DSU` → Scrum framework
- `refinement` → Scrum framework

**Concept-level recommendations:**
- `Verification & Validation (V&V)`: The concept conflates process-level V&V (the topic definition's intent: formal qualification testing, IV&V, design/system verification & validation, V-model gates) with low-level data/model/test verification that pervades ML, ETL, and QA postings. Bare verbs 'verify', 'validate', 'validating', 'validated' supply the bulk of hits and consist almost entirely of non-process matches. → _Drop the bare verb family ('verify','verifies','verifying','validate','validates','validating','validated'). Anchor the concept on the multi-word V&V phrases already in the list ('verification and validation', 'validation and verification', 'V&V', 'IV&V', 'qualification testing', 'design verification', 'system validation', etc.) plus the new 'ASPICE'. Treat single-word 'verification'/'validation' as supporting evidence requiring co-occurrence with 'plan|protocol|design|system|process|qualification' within a small window._
- `Project coordination`: Verb stems 'coordinate', 'facilitate', 'liaise' and their inflections produce 1.5k+ hits each but the snippets show overwhelmingly generic SWE collaboration usage ('coordinate with project leaders', 'facilitate cross-functional collaboration') that any modern engineering JD will contain. The concept currently functions as a SWE-baseline rather than a process-scaffolding marker. → _Replace bare verb forms with phrase-level markers that require an object: 'cross-functional coordination', 'cross-team coordination', 'stakeholder coordination', 'project coordination', 'dependency management', 'stakeholder alignment' (already present). Demote the bare verbs to a 'soft signal' tier counted only when they co-occur with 'sprint|stakeholder|cross-team|dependencies|release' within +/-30 chars._
- `Requirements engineering`: The concept is dominated by the bare token 'requirements' (49% hit rate) which is largely a JD section header, not requirements-engineering practice. The signal-to-noise ratio is poor enough that this concept will appear universal across SWE postings even when no real RE work is described. → _Drop bare 'requirements' and 'requirement' as scoring tokens (or strip the JD-header occurrence in preprocessing). Anchor the concept on action-verb phrases ('gather requirements', 'gathering requirements', 'requirements analysis', 'requirements engineering', 'requirements management', 'requirements traceability', 'translate requirements', 'capture requirements', 'elicit requirements') and tool/artifact names ('DOORS', 'JAMA', 'Polarion', 'BRD', 'SRD', 'NFR', 'RTM', 'traceability matrix', 'use cases'). This preserves the concept's intent without the tautology._
- `Sprints and iterations`: 'velocity', 'iteration', 'iterative', 'epic' all overfire on non-Agile senses (developer velocity as productivity metric, design iteration, Epic Systems healthcare, iterative culture phrasing). The concept is meant to capture time-boxed Scrum/SAFe iteration vocabulary, not generic 'we iterate quickly' culture. → _Restrict to phrase-level: 'sprint*', 'sprint planning/cycle/goal', 'iteration planning', 'team velocity', 'sprint velocity', 'story points', 'user stor*', 'epics' (plural only — singular 'epic' too noisy), 'backlog grooming/refinement', and add 'blockers' / 'DSU' / 'refinement'. Drop bare 'epic', and require co-occurrence guards on bare 'velocity', 'iteration', 'iterative'._
- `Kanban / Lean / Waterfall and other methodologies`: Two of the highest-hit keywords in the concept ('SAFe' 557, 'LeSS' 567) are essentially noise from case-insensitive matching of 'safe' and 'less'. Without case-sensitive matching, the concept's apparent prevalence is fictitious. → _Either enforce case-sensitive matching for these two acronyms (with token boundaries), or require strict co-occurrence guards. Note that 'SAFe' also belongs more naturally in 'Agile methodology' / 'Scrum framework' since it's a scaling framework — consider moving and deduping ('SAFe' already appears under both Agile methodology and this concept)._

### Legacy-stack markers (`legacy_stack`)

The legacy_stack list is generally well-targeted: 216 of 220 keywords fire on this 19,433-posting sample and snippets confirm true legacy-system intent (mainframe, .NET Framework, Java EE, on-prem virtualization). The most concerning weakness is conceptual scope creep on the 'Legacy version control and build' concept: GNU Make and Perforce are not legacy markers (both fire heavily in modern embedded/game-dev contexts), and PL/SQL, while genuinely indicative of Oracle-era DB work, behaves more as an Oracle-DB skill marker than as legacy-vs-modern signal. Several short tokens (Paradox, ADFS, Adabas) need word-boundary or context guards to avoid false positives ('Simpson's paradox', 'Paradox Machines', 'Spark/ADFS/ADF' as Azure data lake variant, 'Adabas' on its own should be paired with Software AG / Natural). The list is missing several obvious legacy markers visible in the snippets themselves: IMS / IMS DB, JEE (bare token), Subversion / SVN / CVS, Crystal Reports, VBScript / ASP 3.0, mainframe utilities (IDCAMS, SMP/E, SDSF, CA-7, Control-M, Autosys), legacy ETL (DataStage), and legacy ESBs (IBM Integration Bus / IIB / DataPower).

_4 drops · 6 guards · 18 adds · 5 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Legacy Java EE / app server stack` — 3.4%
- `Legacy database platforms` — 3.3%
- `Legacy Microsoft server / collaboration` — 2.3%
- `Legacy virtualization and on-prem infrastructure` — 1.7%
- `Legacy identity and directory services` — 1.5%
- `Legacy Microsoft .NET stack` — 1.5%
- `Mainframe languages and platforms` — 1.3%
- `Legacy enterprise integration / ESB / messaging` — 0.8%
- `Legacy enterprise applications and ERPs` — 0.7%
- `Legacy version control and build` — 0.6%
- `Legacy general-purpose languages` — 0.6%
- `Legacy enterprise SOAP / web services` — 0.2%

**Top guards** (false-positive risks worth fixing):
- `Paradox` (Legacy database platforms): Matches 'Simpson's paradox' (statistics term) and the company/product name 'Paradox Machines'/'Paradox' as a startup. 2 of 3 example snippets are false positives. → _guard_: Require a co-occurring DB/legacy context cue: 'Paradox' must appear within ~10 tokens of one of {database, dbase, dBase, Borland, FoxPro, Access, Sybase, Oracle, RDBMS, db}. Drop standalone matches.
- `ADFS` (Legacy identity and directory services): One of 5 example snippets reads 'Spark, ADFS, Databricks, Azure Data Factory' where ADFS appears in a data-platform list, suggesting it may be misused for Azure Data Lake/Factory storage rather than AD Federation Services. → _guard_: Require ADFS to co-occur with one of {Active Directory, AD, Federation, SSO, SAML, identity, authentication, Kerberos, Azure AD, Okta, MFA} within the same posting; otherwise discard.
- `Adabas` (Mainframe languages and platforms): Adabas is a real Software AG mainframe DB, but the bare token risks matching unrelated names. In the sample all 3 hits are valid; concern is forward-looking. Also 'Natural/Adabas' is a redundant separate keyword. → _guard_: Require Adabas to co-occur with one of {Natural, Software AG, mainframe, JCL, COBOL, zOS, z/OS, DB2}. Consider folding 'Natural/Adabas' into the same regex (Adabas\W+|Natural\W*/?\W*Adabas).
- `ESB` (Legacy enterprise integration / ESB / messaging): ESB is a generic architecture acronym; many hits read 'Mule ESB' or 'SOA, ESB concepts' without committing to a heavyweight legacy product. Borderline rather than wrong, but risks inflating concept hit rate. → _guard_: Either rename concept to include 'SOA/ESB architecture (any)' or require ESB to co-occur with one of {TIBCO, WebMethods, BizTalk, IBM Integration, Mule, Oracle Service Bus, JCAPS, WebSphere}. The bare term in a modern microservices JD ('replacing ESB with Kafka') is borderline-legacy and currently included.
- `Perforce` (Legacy version control and build): Perforce/Helix Core remains the de-facto VCS for AAA game studios and large monorepos (Unity/Unreal, embedded). Snippets are dominated by 'Unity/C++ Developer', 'DevOps Engineer building AWS pipelines'. 'Pre-Git legacy SDLC' framing does not fit. → _guard_: Either drop, or restrict to postings where Perforce co-occurs with another legacy VCS marker {ClearCase, PVCS, VSS, CVS, SVN, Subversion} so it is read as part of an enterprise-legacy SCM stack rather than a game-engine workflow.

**Top suggested additions** (grounded in corpus snippets):
- `IMS` → Mainframe languages and platforms
- `IMS DB` → Mainframe languages and platforms
- `IDCAMS` → Mainframe languages and platforms
- `SMP/E` → Mainframe languages and platforms
- `SDSF` → Mainframe languages and platforms

**Concept-level recommendations:**
- `Legacy version control and build`: Concept currently mixes (a) clearly pre-Git legacy SCM (ClearCase, PVCS, VSS, TFVC) with (b) tools that are not 'legacy' in the cloud-native sense — Perforce (still dominant in games/embedded) and GNU Make (still dominant in C/C++/Linux). Mixing inflates the concept hit rate and dilutes the legacy signal. Concept also lacks the most common pre-Git centralized VCS markers (Subversion/SVN/CVS). → _Remove GNU Make. Either remove Perforce or move it to a separate 'specialty / non-cloud SCM' bucket with a co-occurrence guard. Add Subversion, SVN, CVS as primary keywords. Rename the concept to 'Legacy / pre-Git centralized version control'._
- `Legacy database platforms`: Concept mixes truly-legacy DBs (Sybase, Informix, FoxPro, Paradox, DB2/400) with old versions of still-supported DBs (Oracle 10g/11g, SQL Server 2000-2012) and with PL/SQL — which is just 'Oracle DB' and is not by itself a legacy signal. PL/SQL alone is responsible for 41% of the concept's postings_hit (268/648), distorting the rate. → _Split into two sub-concepts: (1) 'Truly legacy DB engines' (DB2 mainframe variants, Sybase, Informix, FoxPro, Paradox, OpenVMS Rdb) and (2) 'Aging-version markers' (Oracle 10g/11g, SQL Server 2000-2012). Move PL/SQL to a guarded 'Oracle ecosystem skill' that only counts when co-occurring with another legacy marker. Add IMS / IMS DB (mainframe hierarchical DB) here or in the mainframe concept._
- `Mainframe languages and platforms`: Strong concept overall, but missing the most common mainframe utilities and schedulers that appear repeatedly in snippets (IDCAMS, SMP/E, SDSF, IMS, IMS DB, CA-7, Control-M, AutoSys, Micro Focus). 'Adabas' and 'Natural/Adabas' are duplicated — current 'Natural/Adabas' (1 hit) is a strict subset of 'Adabas' (3 hits). → _Add IMS, IMS DB, IDCAMS, SMP/E, SDSF, Micro Focus, CA-7, Control-M, AutoSys (latter three could optionally form a 'legacy batch scheduling' sub-cluster). Collapse 'Natural/Adabas' into 'Adabas' with a co-occurrence guard for {Natural, Software AG}._
- `Legacy enterprise SOAP / web services`: Concept overlaps internally with 'Legacy Java EE / app server stack' on JAX-WS and JAX-RPC (cross-list collisions confirm this). The concept is also small (45 postings, 0.23% rate) — it largely captures the same postings as Java EE. → _Either keep JAX-WS/JAX-RPC only in the Java EE concept and remove from this one, or merge this concept into Java EE as 'Legacy Java EE / SOAP services'. Consider that 'SOAP web services' on its own (27 hits) is often a modern integration concern, not legacy — guard with co-occurrence to {WebSphere, WebLogic, JBoss, WS-*, BizTalk, MuleSoft, TIBCO}._
- `Legacy Microsoft server / collaboration`: BizTalk and BizTalk Server are duplicated across this concept and 'Legacy enterprise integration / ESB / messaging'. SSIS/SSRS/SSAS (the SQL Server BI stack) are arguably more 'on-prem MS data tooling' than 'collaboration' — the concept name is misleading. → _Pick one home for BizTalk (the ESB concept is a better fit). Rename this concept to 'Legacy on-prem Microsoft server stack' so SSIS/SSRS/SSAS/TFS fit naturally. Add Crystal Reports and Business Objects (or place those under ERPs)._

### Context infrastructure (`context_infrastructure`)

The list is largely well-targeted but suffers from three recurring failure modes. (1) High-recall observability/data terms fire heavily on industrial-controls and analytics-tooling postings rather than software context infrastructure: 'instrumentation' (222 hits) is dominated by PLC/HMI/field-instrumentation; 'metrics' / 'KPI' / 'KPIs' / 'business metrics' fire on Workday/QA metrics and on-platform reporting unrelated to engineering observability; 'service catalog' is mostly ServiceNow/ITIL; 'APM' is fine but 'application performance monitoring' is reliable. (2) Several keywords are unrelated to the documentation/context concept they are placed under: 'BRD/BRDs' and 'PRD/PRDs' (Architecture decision records) name business/product requirements, not architecture decisions; 'one-pagers' fires on investment decks; 'document processes' is generic procedure-writing; 'playbook'/'playbooks' is heavily Ansible-playbook (config-mgmt) rather than incident playbooks. (3) The 'Cross-functional communication' concept is anchored on extremely broad terms ('stakeholders' 20.5%, 'cross-functional' 17.8%) that match boilerplate JD prose and inflate the concept hit-rate; without a verb-context guard these are essentially noise floors. The list is comprehensive on observability vendors, doc artifacts, and SLI/SLO vocabulary, but needs guards on the broad collaboration terms and pruning of the few mismatched entries.

_14 drops · 12 guards · 14 adds · 5 concept redefines._

**Concept hit rates** (% of sampled SWE postings):
- `Cross-functional communication & coordination` — 36.2%
- `Observability & telemetry stack` — 29.8%
- `Data-pipeline & data-integration hygiene` — 20.7%
- `Technical documentation authoring & maintenance` — 10.9%
- `Technical writing craft` — 9.8%
- `Product & business literacy` — 9.4%
- `Runbooks, playbooks, & operational docs` — 5.6%
- `System understanding & internal knowledge sharing` — 3.7%
- `Architecture decision records & RFCs` — 3.6%
- `Service-level reliability targets` — 2.4%
- `API & interface documentation` — 2.1%

**Top guards** (false-positive risks worth fixing):
- `instrumentation` (Observability & telemetry stack): Industrial / electrical / hardware instrumentation: 'PLCs, motors, pumps, valves, sensors', 'SCADA, instrumentation', 'field instrumentation', 'instrumentation, test, debug, and improve'. Many hits are non-software. → _guard_: Require co-occurrence with software-observability anchor in the same posting: any of {observability, telemetry, metrics, logs, tracing, traces, monitoring, OpenTelemetry, Datadog, Prometheus, Grafana, APM} or use phrase 'code instrumentation' / 'service instrumentation' instead of bare 'instrumentation'.
- `metrics` (Observability & telemetry stack): Generic business/quality metrics: 'business metrics', 'quality metrics', 'KPI metrics', 'security metrics' — not observability metrics. Snippets include 'directly impact key business metrics' and 'Define quality metrics to ensure each step of the testing'. → _guard_: Require co-occurrence with one of {logs, logging, traces, tracing, monitoring, dashboards, observability, telemetry, alerts, alerting} in same posting, OR use bigrams 'application metrics', 'service metrics', 'system metrics', 'operational metrics' (already in list).
- `logs` (Observability & telemetry stack): Bare 'logs' fires on 'application logs and... developer tools', 'audit logs', 'transaction logs' which may indicate debug/forensics rather than the observability stack; also matches in unrelated contexts. → _guard_: Pair-rule: only count when also matching {logging, log aggregation, centralized logging, structured logging, ELK, Splunk, Kibana, Loki, Fluentd, log management} OR keep but rely on co-occurrence with 'metrics'/'traces' in same span (3-pillars pattern).
- `alerts` (Observability & telemetry stack): Workday/Salesforce/Workflow alerts: 'alerts, and Notifications framework', 'Workday Extend... Alerts and Notifications' — these are application-feature alerts, not observability alerts. → _guard_: Require co-occurrence with {alerting, monitoring, on-call, PagerDuty, SLO, SLA, dashboard, incident} in same posting.
- `dashboard` (Observability & telemetry stack): BI / Tableau / data-analyst dashboards: 'Tableau Dashboard development', 'dashboard formats, visualization style', 'dashboard creation, deep understanding of user interface' — these are analytics/BI dashboards, not ops dashboards. → _guard_: Require co-occurrence with ops anchor {monitoring, alerting, observability, Grafana, Datadog, on-call, SLO, metrics, telemetry} OR drop in favor of 'dashboards' (plural is more often ops-flavored, e.g. 'Splunk dashboards').

**Top suggested additions** (grounded in corpus snippets):
- `Notion` → Technical documentation authoring & maintenance
- `internal knowledge bases` → Technical documentation authoring & maintenance
- `design documents` → Technical documentation authoring & maintenance
- `blameless postmortems` → Runbooks, playbooks, & operational docs
- `incident response` → Runbooks, playbooks, & operational docs

**Concept-level recommendations:**
- `Architecture decision records & RFCs`: Concept is mixing engineering decision artifacts (ADRs, RFCs, tech specs, design proposals) with upstream business/product requirements artifacts (BRD, PRD, one-pagers, product requirements document). The latter belong in 'Product & business literacy' or in the process_scaffolding topic; they describe WHAT to build, not the engineering decision about HOW. → _Drop BRD, BRDs, PRD, PRDs, product requirements document, one-pagers from this concept. Optionally move them to Product & business literacy (or coordinate with process_scaffolding owner). Tighten definition to 'engineering-authored decision artifacts (ADRs, RFCs, tech specs, design proposals, architecture reviews)'._
- `Cross-functional communication & coordination`: Anchored on two ultra-high-recall terms ('stakeholders' 20.5%, 'cross-functional' 17.8%) that fire on JD boilerplate in nearly every senior-IC posting. The 36.2% concept hit-rate is largely floor noise rather than signal about cross-functional communication craft. Also has both hyphenated and unhyphenated forms ('cross-functional' vs 'cross functional', 'cross-org' vs 'cross-organizational' vs 'cross-organization' vs 'cross org' vs 'crossfunctional') that collectively over-count. → _(1) Apply verb-anchor guards to 'stakeholders' and 'cross-functional' (see guards). (2) Consolidate hyphenation variants into a single normalized regex per surface form rather than counting each as an independent keyword. (3) Either lower the weight of these two terms or move them to a distinct 'team-shape' concept distinct from 'translation/communication craft' (which is what the concept definition emphasizes)._
- `Observability & telemetry stack`: Mixes high-recall ambiguous terms ('monitoring', 'metrics', 'logs', 'alerts', 'dashboard', 'instrumentation', 'tracing', 'traces') with vendor names that are unambiguous. The single-token entries pick up large numbers of false positives from BI dashboards, business metrics, Workday alerts, hardware instrumentation, and requirements tracing. → _Either (a) require an observability-anchor co-occurrence guard for each ambiguous single-token (see guards), or (b) restrict single-token entries to plural/bigram forms that are more discriminative ('dashboards' over 'dashboard', 'distributed tracing' instead of 'tracing'). Vendor list is in good shape — extend with Sentry, Jaeger, Dynatrace, SumoLogic, Fluentd._
- `Runbooks, playbooks, & operational docs`: 'playbook'/'playbooks' is dominated by Ansible config-management playbooks (a code artifact for declarative provisioning), not the operational/incident playbooks the concept targets. This injects systematic FPs into an otherwise clean concept. → _Add an Ansible exclusion guard for 'playbook' and 'playbooks'. Add 'incident response' and 'PagerDuty' as direct positive signals (see adds)._
- `API & interface documentation`: 'service catalog' is ITSM/ServiceNow terminology, not API documentation; 'DX' is a two-letter token with extreme ambiguity; 'schema documentation' is duplicated across this concept and Data-pipeline & data-integration hygiene (the only within-topic duplicate). → _Drop 'service catalog' and 'DX' from this concept. Resolve the 'schema documentation' duplicate by keeping it only in API & interface documentation (the single observed snippet is GraphQL schema documentation, which fits API docs better than data-pipeline hygiene). Add 'Backstage' as a direct positive signal for internal developer portals._

## Cross-list reconciliation

`collisions.json` lists every keyword appearing in ≥2 topics. The patterns below cluster the most common collisions and propose a canonical home for each.

**Top topic-pair collision counts:**

| Topic A | Topic B | # collisions |
|---|---|---:|
| context_infrastructure | verification | 36 |
| orchestration | process_scaffolding | 16 |
| context_infrastructure | orchestration | 12 |
| process_scaffolding | verification | 12 |
| context_infrastructure | process_scaffolding | 11 |
| performance | verification | 10 |
| mentorship | verification | 10 |
| orchestration | verification | 8 |
| context_infrastructure | mentorship | 8 |
| mentorship | people_management | 7 |

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
