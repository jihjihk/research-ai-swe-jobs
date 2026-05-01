# Artifact 3 — Senior Applied-AI / LLM Engineer archetype exemplars (T34 cluster 0)

**Source:** T34 cluster 0 profiling. 20 representative 2026 postings from `exploration/tables/T34/content_exemplars_cluster0.csv`. Cluster 0 n=2,395 total (144 in 2024 + 2,251 in 2026 = **15.6× growth**). 94% of cluster 0 is 2026-dated.

**THIS IS THE SINGLE MOST IMPORTANT ARTIFACT FOR RQ4 INTERVIEWS.** Show real posting text to senior engineers and ask: *"Is this a real job or a rebranding?"*

**Cluster 0 fingerprint (from T34):**
- ai_binary = 1.00 (by construction)
- orch_density = 1.86 / 1K chars (high, only slightly below cluster 1's 2.27)
- mgmt_density_v1_rebuilt = 0.001 (near zero; AI-oriented seniors are NOT people managers)
- Median YOE = **6.0 years** (1 year MORE than cluster 1's 5.0)
- Director share = 1.9% (2× cluster 1's 1.0%)
- Industry: Software Development 44.6%, Financial Services 16.5%, IT Services 13.6%
- Titles: 28.2% "AI Engineer" (true share ≥45% per T34 manual spot-check), 17.8% senior engineer, 8.0% staff engineer, 6.0% tech lead

**Distinguishing bigrams (log-odds +4.5 to +5.5):** `claude code`, `rag pipelines`, `langchain llamaindex`, `rag architectures`, `copilot claude`, `github copilot claude`, `cursor claude code`, `augmented generation rag`, `guardrails model`, `training large language`.

---

## Exemplar A — The Phoenix Group (Senior Artificial Intelligence Engineer)

> **Industry:** Business Consulting, IT Services, Financial Services | **YOE:** 8 years
>
> "**Core Responsibilities** — Data Platform & Architecture: Design and implement cloud-based data lake / warehouse architectures centered around modern platforms (e.g., Snowflake). Develop data ingestion pipelines from enterprise systems (project management, ERP, HR systems, contract databases). Build structured and unstructured ingestion workflows using Python, REST APIs, ETL tools, and serverless..."
>
> Orch density: 12.0 / 1K chars (very high) | Mgmt density: 0.0 | AI binary: 1

## Exemplar B — Coforge (Senior AI Engineer, Generative AI & Agentic Systems)

> **Industry:** IT Services and IT Consulting, Investment Banking | **YOE:** 12+ years
>
> "Coforge is seeking a highly skilled Senior AI Engineer to design, build, and deploy Generative AI and Agentic Systems. Role focuses on production LLM integration, RAG architectures, and multi-agent orchestration at enterprise scale..."
>
> Orch density: 10.1 / 1K chars | Mgmt density: 0.0

## Exemplar C — Menos AI (Machine Learning Engineer — Financial Services)

> **Industry:** Financial Services | **YOE:** 3 years
>
> "As a Machine Learning Engineer at Menos AI, you will design, build, and optimize the specialized AI agents that power our research intelligence platform for institutional asset managers. Our platform processes large-scale financial data to deliver actionable insights... You will work across multi-LLM pipelines (Anthropic, OpenAI, Gemini), agentic orchestration..."
>
> Orch density: 8.3 / 1K chars | Cluster 0 exemplar despite relatively junior YOE

## Exemplar D — Clear Street (Senior Software Engineer — Studio — Java, AI)

> **Industry:** Financial Services | **YOE:** 7 years
>
> "As a Senior Software Engineer, you'll build the backend that powers AI features in our trading platform (Clear Street Studio). You'll own the AI service layer: retrieval, orchestration, guardrails, and evaluation. We're looking for people who take ownership, learn quickly, and deliver without heavy oversight in a fast-moving environment."

## Exemplar E — Emergence AI (Staff AI Agent Engineer)

> **Industry:** Technology, Information and Internet | **YOE:** 10 years
>
> "We're seeking a Staff/Principal Agentic AI Engineer to define and own the production architecture for our autonomous agent platform. You'll build reliable, safe, and scalable systems that power complex agentic workflows, combining deep AI/ML expertise with systems engineering to create production-grade enterprise agents. Build Production AI Agent Systems..."

## Exemplar F — NTT DATA North America (Lead MLOps Engineer)

> **Industry:** IT Services and IT Consulting | **YOE:** 8 years
>
> "As a professional in this role, you will own the infrastructure and operationalization of machine learning and Generative AI systems at scale. This includes building reliable pipelines for training, evaluation, deployment, and monitoring of models in production environments. What You'll Do — ML/LLMOps Architecture: Design end-to-end pipelines for training and deploying models, including LLM..."

## Exemplar G — Octans Group (Senior AI Full Stack Engineer, LLM/GenAI)

> **Industry:** Software Development | **YOE:** 5 years | Title: "Onsite Interview | Sr. AI Full Stack Engineer (LLM / GenAI) | Bay Area"
>
> "We are hiring a Senior AI Full Stack Engineer to build real-world AI-powered applications using LLMs and Generative AI. This role focuses on end-to-end development of scalable AI solutions integrated into business workflows..."

## Exemplar H — Kai (Senior Data Engineer, AI Platform)

> **Industry:** Computer and Network Security | **YOE:** 4 years
>
> "We are looking for a Senior Data Engineer (AI Platform) to design and build scalable data systems that power next-generation AI and Generative AI applications. This is a senior, hands-on technical role for someone who can operate across both classical data engineering and modern AI data infrastructure — including large-scale data pipelines, vector databases, and retrieval system..."

---

## What the 20 exemplars have in common

Across the sampled 20 cluster-0 postings, recurring phrase patterns appear in 12+ of 20:

- "LLM / RAG / prompt engineering" — 12/20
- "pipelines" (data or ML pipelines) — 13/20
- "agentic / multi-agent" — 8/20
- "orchestration / workflow" — 7/20
- "system design / architecture" — 7/20
- "production / production-grade" — 9/20
- "vector database" or "retrieval" — 6/20
- "evaluation / guardrails" — 3/20
- "fine-tuning / training" — 1/20 (notably low — LLM customization is not a majority ask)

Tools named:
- LangChain — 5/20
- LlamaIndex — 3/20
- OpenAI API — 4/20
- Anthropic / Claude — 3/20
- Gemini — 2/20
- Hugging Face — 2/20
- GitHub Copilot — 2/20 (surprisingly low given T23 overall rate)

---

## Why this is the most important interview artifact

The T34 cluster-0 archetype is a **content-coherent, 15.6×-growth, independently-identifiable** emergent senior role. It is NOT simply "ML Engineer" rebranded (ML Engineer was a 2024-established title; cluster 0 titles include "AI Engineer", "AI Platform Engineer", "Applied AI Engineer", "Agentic AI Engineer" — most did not exist at scale in 2024). The median YOE (6 years, 1 year HIGHER than cluster 1) and 2× director share suggest the archetype is being populated by senior people, not by lateral re-titling of juniors.

**Three competing interpretations that interviews must disambiguate:**

1. **(a) Genuine new role.** The 2026 Applied-AI Engineer builds production LLM systems with RAG + agentic orchestration + guardrails. This is meaningfully different from 2024 ML Engineer (research-adjacent, model-training-focused). Supported by: 1,163 firms posting cluster-0 roles (not a single-firm artifact), T09 archetype over-representation 6.75× in models/systems/llm cluster.

2. **(b) Rebranding of senior SWE.** Employers renamed existing senior software engineers to AI Engineer as an HR-signaling move. The actual work unchanged. Supported by: 17.8% of cluster 0 titles are "senior engineer" not "AI engineer"; Clear Street's "Senior Software Engineer - Studio - Java, AI" is a senior SWE with AI framing.

3. **(c) Skills-stacking of existing ML Engineer role.** What was "ML Engineer" in 2024 absorbed LLM/agentic/RAG skills in 2026. The title and seniority boundary are the same; the skillset broadened. Supported by: cluster 0 median YOE 6.0 overlaps with ML engineer YOE bands; T18 found ML engineer AI-strict crossed 50% in 2026.

## Interview questions for senior engineers

1. **Show exemplars A, B, E, F, G.** "Is this a role you'd take? Is it one you'd hire for? Is this a distinct role from 'Senior Software Engineer' at a non-AI-forward firm?"

2. **Ask about the 2024 analog.** "What was the closest 2024 equivalent role? Was it 'ML Engineer'? 'Senior Backend Engineer'? Something else?"

3. **Ask about seniority asymmetry.** "The data shows AI-oriented senior postings ask for MORE experience (median 6 years) than traditional senior SWE postings (5 years). Does this match your experience? Why might AI roles be MORE senior-gated?"

4. **Ask about the director-share finding.** "Director-level postings in this archetype grew from 1.1% to 14.8% of senior director postings. Is 'AI-focused director' a real role, or is director-title being used loosely in AI postings?"

5. **Ask about financial-services concentration (17% of cluster 0).** "Why might banks and insurers be over-represented in Applied-AI senior hiring? Regulatory? Talent-market? Scale?"

## Attribution / anonymity note

Specific firm names (Menos AI, Emergence AI, Clear Street, Kai, Coforge) should be anonymized in interview materials per protocol. The archetype-level finding stands independently of any single firm attribution.
