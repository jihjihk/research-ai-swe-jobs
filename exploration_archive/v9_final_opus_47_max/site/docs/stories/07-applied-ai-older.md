# The democratiser that asks for more experience

### *Large-language-model tooling is said to be compressing the experience premium in software. The single role most identified with the technology demands more experience, not less.*

In January 2023 Andrej Karpathy, formerly of Tesla and OpenAI, tweeted to four million eventual viewers that "the hottest new programming language is English." Jensen Huang of Nvidia, speaking at a conference in February 2024, advised teenagers not to bother learning to code: the programming language was now human. Sam Altman of OpenAI said the same thing a month later in more words: "coding will go away." Matt Welsh of the *Communications of the ACM* had already written, in December 2022, a piece titled "The End of Programming." The Karpathy–Huang–Altman consensus was that language models would end programming as a gated skill. What remained was English prompts; anyone with a good one could ship.

The consensus is about *access to programming*, not about the specific hiring bar for people who work on AI systems themselves. That distinction matters, because the two things are moving in opposite directions in the posting data.

Of 33,693 senior software-engineering postings partitioned into five content-based clusters, one cluster is unmistakably the AI-role archetype: its distinguishing phrases are "claude code", "rag pipelines", "github copilot claude", "langchain llamaindex". The cluster grew from 144 postings in 2024 to 2,251 in 2026, a 15.6-fold rise; 94% of its volume is 2026. It is the role most emblematic of the Karpathy thesis. It is where, if language-model tooling were collapsing the experience premium *for AI work itself*, the collapse would be most visible.

Median years of experience asked by this cluster is 6.0. The median for the other senior clusters is 5.0, a one-year gap on a base where, at senior level, a year is a meaningful premium. The cluster's share of directors is 1.9%, against 1.0% in the comparator cluster, a 0.9-percentage-point gap on a base of roughly 2% (statistically modest in absolute terms but a nearly doubled proportional share of directors). The cluster is not a niche either. Its Herfindahl-Hirschman index (a concentration measure: 0 is evenly distributed, 100 is single-firm dominance) across 1,163 distinct firms is 38.6, well away from concentration. 45% of its top-10-industries share comes from software development; 17% from financial services.

A typical Senior Applied-AI / LLM Engineer posting in the 2026 sample asks for a bachelor's or advanced degree, five to eight years of engineering experience, two of them with production large-language-model systems, direct experience fine-tuning models or building retrieval-augmented-generation pipelines, and facility with at least three of LangChain, LlamaIndex, Pinecone, vector databases and Claude Code. Fourteen of twenty sampled titles from the cluster were explicitly titled "AI", "ML", or "Machine Learning" engineer at senior or above; one was "Sr. Distinguished AI Engineer". None is entry-level.

The democratisation thesis may well be true on the dimension its proponents meant, access to programming. Fastly's research on AI-assisted development finds that senior engineers ship 2.5× more AI-assisted code per day than junior engineers, not because their skill has doubled but because their pattern-recognition makes the tool's output useful. A tool that amplifies existing skill can also democratise the bottom of the distribution. But the *credential bar* for the AI-specialist role itself has moved upward. Gergely Orosz, writing in *Pragmatic Engineer* in 2025, noted the same phenomenon from the hiring-manager side: firms paying for production AI systems now insist on experience many juniors simply cannot have accumulated.

Industry rhetoric and hiring-manager practice are running in opposite directions. When a CEO says AI lowers the bar, that is a product-marketing claim. When a recruiter writes "five to eight years, two with production LLMs," that is a budget commitment. Budgets override rhetoric.

The closing image belongs to the counterfactual. If language-model tooling really were collapsing the experience premium on work that most resembles what the tools do, the Senior Applied-AI / LLM Engineer posting of 2026 would be three-to-four years of experience, heavy on prompt fluency, short on system-design depth. The one in the data is six years, architecturally senior, and budgeted for accordingly. Whatever AI is doing to the broader software-engineering ladder, it is not collapsing the skill ceiling on the job of building AI itself.

---

??? note "Evidence and sources"

    **Headline numbers**

    - Senior Applied-AI / LLM Engineer cluster grew 15.6× (144 → 2,251 postings); 94% of postings are 2026. T34 cluster 0; fact-check 07 verified exactly.
    - Median YOE 6.0 (cluster 0, n=1,511 labeled) vs 5.0 (cluster 1, n=5,550 labeled).
    - Director share 1.921% (cluster 0) vs 1.015% (cluster 1); ratio 1.89×.
    - Top-10-industry normalized mix: 44.6% Software Development + 16.5% Financial Services + 13.6% IT Services. (Denominator is cluster-0 top-10 industry rows at n=1,445, not total 2,251 cluster rows; raw Software Development share across all cluster-0 rows is 28.6%.)
    - 1,163 distinct firms; HHI 38.6.
    - Distinguishing bigrams: "claude code" (+5.52), "rag pipelines" (+5.04), "github copilot claude" (+5.02), "langchain llamaindex" (+4.86), "augmented generation rag" (+4.72).
    - Cluster 0 rise robust across T30 senior-definition panel (S1 / S2 / S3 / S4, all 10-14× rise each) — V2 Phase C.
    - Content coherence: 14 of 20 sampled cluster-0 titles explicitly "AI", "ML" or "Machine Learning" — V2 Phase B.
    - Fastly research: senior engineers ship 2.5× more AI-assisted code per day than juniors. Fastly engineering blog, 2025.
    - Orosz "Seniority Rollercoaster" — *Pragmatic Engineer*, 2025 (counter-voice cited in body).

    **Conventional-wisdom opponent**

    Andrej Karpathy's January 2023 tweet ("the hottest new programming language is English"), with 4M views; Jensen Huang's February 2024 "don't learn to code"; Sam Altman's March 2024 "coding will go away"; Matt Welsh "The End of Programming" (*CACM*, December 2022). The piece distinguishes between their claim (access to programming) and the narrower finding (hiring bar for AI-specialist roles).

    **Sensitivity verdict**

    Cluster 0 identification is via a k=5 k-means on senior cohort; silhouette 0.477. Cluster definition is robust across four T30 senior-definition variants and aggregator exclusion (< 3% metric shift). Title-level content coherence confirmed on 14 of 20 sampled titles. The "older" finding is a median comparison on labeled subset (n=1,511 cluster 0 vs n=5,550 cluster 1). The 44.6% industry share is normalized within the top-10 industry rows of cluster 0 — the evidence block states this explicitly; the raw unnormalized Software Development share across all cluster-0 rows is 28.6%. The director-share "2×" is a 0.9 pp gap on a 2% base, statistically modest but proportionally meaningful.

---

## Related in Findings

- [Two new senior archetypes appear in 2026: "applied AI" and "forward-deployed"](../findings/a5-archetypes.md) — the two-archetype profiling that cluster 0 (Applied-AI) comes from.
- [Junior and senior job descriptions moved apart between 2024 and 2026, not together](../findings/a3-seniority-sharpening.md) — the broader context of within-seniority distinctness rising, not falling.
