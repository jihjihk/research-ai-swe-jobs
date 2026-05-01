---
title: "New angles from technical commentators / substack / developer discourse"
scope: tech-commentator
claims_count: 8
date: 2026-04-21
---

# New angles — practitioner discourse (tech commentators, substack, HN, SO)

Eight testable claims from 2024-2026 practitioner-facing discourse. All go beyond the 8 existing pieces. Signal HIGH/MED marked.

---

## 1. Forward-Deployed Engineer is the breakout archetype (+800%) [HIGH]
- Src: Interview Query / bloomberry, Oct 2025 (https://www.interviewquery.com/p/ai-forward-deployed-engineer-jobs-2025, https://bloomberry.com/blog/i-analyzed-1000-forward-deployed-engineer-jobs-what-i-learned/); PYMNTS Jan 2026.
- Claim: Monthly FDE postings grew >800% Jan-Sep 2025; Anthropic/OpenAI/Palantir/Cohere/Salesforce dominant; avg $174K.
- Data-led.
- Test: archetype/title regex for "forward deployed", "solutions engineer", "deployment engineer" across 2024 vs 2026; YoY multiple by named AI-lab employers.
- Orthogonal to all 8 pieces.
- Contrarian headline: "The fastest-growing SWE job of 2025 isn't an engineer — it's a customer."

## 2. Prompt-engineer titles collapse while RAG/LangChain surges [HIGH]
- Src: Salesforce Ben Apr 2025 (https://www.salesforceben.com/prompt-engineering-jobs-are-obsolete-in-2025-heres-why/); Fortune May 2025 (https://fortune.com/2025/05/07/prompt-engineering-200k-six-figure-role-now-obsolete-thanks-to-ai/); Flex.ai state-of-AI-hiring 2025.
- Claim: "Prompt engineer" titles dropped ~40% mid-2024 to early-2025; RAG appears in ~2k postings; LangChain in >10% of AI JDs. Skill survived as feature, not role.
- Data-led.
- Test: title regex "prompt engineer" YoY vs tech-mention tokens for rag/langchain/langgraph/vector db; expected inverse curve.
- Orthogonal.
- Headline: "Prompt engineering died so the tech stack could live."

## 3. Frontend contracts faster than backend under AI [HIGH]
- Src: ITCompare frontend-vs-backend age-of-AI (https://itcompare.pl/en-us/articles/77/frontend-vs-backend-in-the-age-of-ai-who-wins-and-who-loses-by-2026); gocodeo/AvidClan/Nanobyte 2025 — "biggest drop in frontend demand" in 2025; 40% of routine FE tasks automated.
- Claim: Frontend-archetype postings drop meaningfully more than backend 2024-2026; delta widens inside AI-positive postings.
- Data-led (loose) — sharpen to our data.
- Test: archetype FE vs BE share-change x AI-pattern flag.
- Shape-agrees with 01_narrowing_middle but adds role-axis absent there.
- Headline: "AI came for the div before it came for the database."

## 4. TypeScript/React oligopoly tightens under AI [MED]
- Src: IT Support Group 2026 guide (https://thisisanitsupportgroup.com/blog/best-programming-languages-2026-complete-guide/); careerproguider 2025 React-v-Angular.
- Claim: >80% of FE postings require TypeScript (vs ~50% in 2023); React holds ~50% share. AI copilots reinforce TS/React because training data is TS/React-heavy.
- Data-led.
- Test: tech-mention share of typescript, tsx, react, next.js in FE postings 2024 vs 2026 vs Vue/Svelte/Angular.
- Orthogonal to 05 (unvalidated regex) but adjacent.
- Headline: "The AI-copilot era killed JavaScript diversity, not JavaScript itself."

## 5. Rust/Go are the quiet senior-favored corridor [MED]
- Src: Wojtczyk HN-Hiring trend, Feb 2025 (https://martin.wojtczyk.de/2025/02/20/rust-c-and-python-trends-in-jobs-on-hacker-news-february-2025/); HN `Who is hiring?` (https://hnhiring.com/technologies/rust); Acceler8 2025 Rust salaries.
- Claim: Rust postings tripled in 2 yrs; Rust pays 15-20% premium over Go/Python; Rust/Go have lower junior share than Python/JS.
- Data-led.
- Test: tech-mention Rust/Go/Python x seniority distribution; YoY growth ratios.
- Agrees with 07_applied_ai_older (senior skew) on a language axis.
- Headline: "Rust is the senior-engineer lounge of 2026."

## 6. CLI/terminal-native coding tools are a senior signal [MED]
- Src: Batsov "Emacs and Vim in the Age of AI", Mar 2026 (https://batsov.com/articles/2026/03/09/emacs-and-vim-in-the-age-of-ai/); Orosz "Pragmatic Engineer in 2025" on Claude Code CLI (https://newsletter.pragmaticengineer.com/p/the-pragmatic-engineer-in-2025), Dec 2025.
- Claim: postings naming Claude Code / Cursor / Aider / Windsurf cluster at senior/staff + Applied-AI archetype; Vim/Emacs mentions not declining.
- Mixed data + speculative.
- Test: regex named-tools vs seniority; compare with generic "copilot" clusters.
- Sharpens 02_copilot_paradox — does aggregate 0.10% hide tool-family clusters?
- Headline: "If the JD names the tool, ask which terminal."

## 7. Mid-career squeeze, not junior apocalypse [HIGH]
- Src: Stanford Digital Economy via Stack Overflow "AI vs Gen Z", Dec 2025 (https://stackoverflow.blog/2025/12/26/ai-vs-gen-z/); Willison 2025 year-in-LLMs, Dec 31 2025 (https://simonwillison.net/2025/Dec/31/the-year-in-llms/) — explicitly flags MID-career as most at risk post-Nov 2025.
- Claim: devs 22-25 fell ~20% since 2022 peak; Willison says mid-career (3-7 YOE) is the new vulnerable middle.
- Data-led for juniors; named speculation for mid-career.
- Test: seniority distribution 2024 vs 2026 with explicit mid-career bucket — is 3-7 YOE thinning faster than entry?
- Potentially contradicts direction of 01_narrowing_middle — strongest contrarian candidate.
- Headline: "Willison was right — it's the middle, not the bottom, that broke in 2025."

## 8. AI-builder geography stays hub-locked even as AI-user geography uniforms [HIGH]
- Src: bloomberry 1000-FDE analysis 2025 (https://bloomberry.com/blog/i-analyzed-1000-forward-deployed-engineer-jobs-what-i-learned/); LangChain state-of-agent-engineering (https://www.langchain.com/state-of-agent-engineering); Byteiota remote-AI 2026 (https://byteiota.com/remote-developer-jobs-2026-ai-ml-roles-soar-generalists-face-rto/).
- Claim: FDE/agent-engineer postings disproportionately on-site/hybrid in SF/NYC/SEA; general SWE remote ~27%. AI-BUILDER subclass is less remote and more concentrated — the opposite of the copilot-user uniformity.
- Data-led.
- Test: filter archetype in (forward-deployed, agent, applied-ai) → metro + remote flag vs full panel.
- Disagrees with piece 03 (Atlanta-not-SF) — uniformity holds for copilot-users, breaks for AI-builders.
- Headline: "The AI geography IS uniform — except where AI is actually built."

---

## Background-only (flagged LOW, not claims)
- Founding-engineer market (Dover 2025) — slice too small.
- AI red-teamer +55% (AICareerFinder 2025) — small absolute n; probe not lead.
- Karpathy "Software 3.0" / Ben Thompson-Truell interview (stratechery.com/2025, Jun 2025) — framing, not testable.
- PG "Writes and Write-Nots" / Founder Mode — skip.

## Piece priorities
1. #1 FDE +800% — cleanest novel quant.
2. #7 Mid-career squeeze — contrarian vs piece 01.
3. #8 AI-builder hubs — sharpens piece 03.
4. #2 Prompt-eng dies / RAG lives — clean inverse.
5. #3 FE-before-BE — if archetype data clean.
6-8. #4/#5/#6 — supporting MED.
