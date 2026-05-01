# Artifact 1 — Microsoft "Software Engineer II" pair-level rewrite

**Source:** T31 pair-level drift analysis. Pair: `company_name_canonical='Microsoft'`, `title_lc='software engineer ii'`. Panel: `arshkon_min3_n3` and `pooled_min5_n3`.

**Pair-level statistics:**
- 2024 n: 6 postings
- 2026 n: 35 postings (5.8× growth in volume at this specific title within the same firm)
- Δ ai_strict_v1: **+0.400** (40% of 2026 postings mention strict AI tokens; 0% in 2024)
- Δ requirement_breadth_resid: +0.77 (scope broadened marginally within-title)
- Δ reqs_share_t13: −0.018 (requirements section shrank slightly as share of total)
- Δ desc_length: +342 chars (descriptions modestly longer)
- is_aggregator: False (direct employer)

---

## 2024 Microsoft "Software Engineer II" representative content (pre-AI-rewriting)

*Excerpt from exploration-phase reads of 2024 Microsoft SE II postings; representative theme.*

The 2024 postings emphasized:
- Generic cloud-platform work ("Azure services", "cloud-scale backend")
- Distributed-systems engineering ("microservices", "distributed data processing")
- Cross-team collaboration ("partner with product managers and designers")
- Traditional SDE competencies ("object-oriented programming", "data structures", "algorithms")

**NO mentions of:** Copilot, GitHub Copilot, Generative AI, LLM, RAG, Claude, AI Systems org, AI Foundry.

---

## 2026 Microsoft "Software Engineer II" representative content (post-AI-rewriting)

*Excerpt from T31-saved top-AI-drift Microsoft Software Engineer II postings, n=35 at +40% ai_strict rate.*

Recurring content themes in the 2026 cluster:

### Theme A — Copilot and Generative AI as product surface

> "You'll contribute to the next generation of GitHub Copilot features by building backend services that power our AI-assisted developer experiences. You'll partner with researchers on prompt engineering, RAG pipelines, and AI agent orchestration for production coding scenarios."

### Theme B — AI Systems / AI Foundry org identification

> "This role sits within the AI Systems org at Microsoft Cloud & AI. You'll work alongside applied scientists and AI engineers to ship Generative AI features that are used by millions of developers daily."

### Theme C — AI engineering skills mentioned alongside traditional SDE skills

> "Familiarity with modern AI engineering practices (prompt engineering, RAG, evaluation frameworks, fine-tuning) is a strong plus. You should be comfortable writing production code in C# / Python / TypeScript and partnering with ML infrastructure teams."

### Theme D — Cloud-infrastructure + AI-service integration

> "Build and operate distributed services that serve Generative AI inference workloads at scale. Partner with the AI Foundry team on platform infrastructure for foundation-model-as-a-service offerings."

---

## Why this pair is an interview prompt

1. **Microsoft's "Software Engineer II" is a standardized ladder rung.** The title has not changed between 2024 and 2026. Yet the CONTENT has shifted substantially — from generic cloud/platform to AI-product surface area.
2. **+40 pp AI-mention rise at pair level** exceeds the aggregate within-firm rewriting signal (+7.7-8.3 pp at company level per T16). The Microsoft SE II rewrite is one of the largest documented at n≥6-per-period.
3. **YOE and credentials are stable** at this title (per T31 aggregate findings; rule-YOE median drops ~0.3-1.0 yr at pair level but Microsoft SE II is specifically title-standardized). The rewrite is about WHAT the role does, not WHO can apply.

## Interview questions

1. **To Microsoft hiring managers / tech-lead informants.** "Did Microsoft's SE II role fundamentally change between 2024 and 2026, or did the description language catch up to work that was already happening? Was the rewrite organization-driven (centrally re-templated) or team-driven (individual teams updating their JDs)?"

2. **To senior IC engineers at Microsoft Cloud/AI org.** "Does the 2026 SE II JD accurately describe your day-to-day? What fraction of SE II engineers at Microsoft actually work on Copilot / AI Systems vs. generic cloud infrastructure?"

3. **To recruiters / talent-acquisition partners.** "How much of the 2026 JD language came from: (a) an LLM-based JD rewriting tool, (b) direction from leadership to emphasize AI, (c) the actual hiring teams' specific need for AI-experienced candidates? Can you rank these mechanisms?"

4. **To candidates who applied to Microsoft SE II roles across 2024 and 2026.** "If you saw the 2024 and 2026 JDs side-by-side, would you describe them as the same role or different roles? Did you prepare differently in 2026?"

## Attribution caveat

Microsoft is a single firm and appears in T31's top-20 AI-drift exemplars with the largest absolute n (35 postings in 2026). Generalizing from Microsoft to "all large US tech firms" is justified only by the T31 finding that Wells Fargo, Capital One, Amazon, and Walmart show similar pair-level AI rewriting patterns. Interviewees may recognize the specific Microsoft example; anonymize per protocol.
