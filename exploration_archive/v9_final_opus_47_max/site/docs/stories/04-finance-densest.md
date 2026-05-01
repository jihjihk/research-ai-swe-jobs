# What the compliance lobby can't see

### *Banks and insurers are supposed to be the AI laggards. Their job descriptions do not look it.*

The received wisdom is emphatic. Deloitte's 2026 Banking & Capital Markets Outlook describes an industry "throttled by… mounting compliance demands" on artificial intelligence. McKinsey's banking team writes about "AI angst" and "pilot purgatory." Gartner tells subscribers that 73% of banking AI pilots never reach production. A 2025 industry adoption report finds a thirty-four-point gap: 92% of technology firms using AI against 58% of regulated firms. The framing, from *American Banker* to the *Financial Times*, is that compliance and caution have left regulated industries behind.

The job advertisements tell a different story.

Among 25,547 software-engineering postings collected from LinkedIn across the first half of 2026 (each measured against a 96%-precision pattern for artificial-intelligence requirement language), the overall prevalence of an explicit AI-tool or AI-skill requirement is 13.24%. Financial Services posts 15.74% (n=1,785). Software Development posts 15.25% (n=6,870). The two rates are statistically indistinguishable: their 95% Wilson confidence intervals (a standard interval estimate for proportions; overlapping intervals mean the difference isn't statistically distinguishable at the 95% level) overlap substantially. Regulated finance is not ahead of the software industry on AI codification, but it is not behind it either. Both sectors sit clearly and significantly above the 13.24% baseline, whose confidence interval does not overlap with either.

Nor only finance. Insurance comes in at 15.0%. Banking at 14.5%. And, the most uncomfortable number for the "regulated sectors lag" narrative, hospitals and health care (a sector the compliance commentary treats as even more restricted than banking) come in at 20.87%, higher than technology. Strip the small-n outlier cells away and the picture is of a labour market in which the sectors the consultancy reports describe as dragging their feet are, on the contrary, writing the same kind of AI-requirement language that software firms are.

The named protagonists make the point concrete. Citigroup advertises a "Gen AI-ML Engineer, AVP" whose description names RAG, LlamaIndex, LangChain, Gemini, GPT-4, prompt engineering, fine-tuning, and vector databases: more than five AI-tool mentions per thousand characters of description, one of the densest requisitions in the entire dataset. JPMorgan Chase's "Software Engineer III" lists fine-tuning, RAG, prompt engineering, and Hugging Face. Visa advertises for a "Staff Software Engineer, AI Agent/Agentic Focus" building what the posting calls a "next-generation agentic AI platform" that integrates Claude, Gemini, Pinecone and LangChain. GEICO posts for a senior applied-AI engineer in Java. BlackRock's software-engineering postings carry an AI-requirement rate of 44.4% when filtered to reasonable-volume cells. Wells Fargo is at 35.7%; SoFi at 33.3%.

There are two ways to reconcile the consultancy reports with the posting data. The first, charitable reconciliation is that the narratives measure different things. Deloitte, McKinsey and Gartner are writing about production deployment, pilot-to-scale conversion, and governance maturity. The posting data measures what firms are willing to ask of a new engineer in 2026. By the first measure, regulated firms may genuinely trail technology firms. By the second, they are not trailing.

The second reconciliation, less comfortable for the consultancy authors, is that the narrative of regulated-industry caution is priced by compliance lobbyists, by management consultants selling transformation projects, and by vendors whose business model requires the sector to appear behind. The firms themselves, when recruiting, write as though they were competing for the same talent as Anthropic.

What this evidence cannot decide is whether regulated firms eventually run the systems their job descriptions demand, or whether the job descriptions will stay ahead of the production environment for the next two years. The compliance friction that the consultancies describe could yet produce a gap between what gets hired for and what gets deployed. What it has not produced, in the posting data, is a gap in how the hiring is written.

Banks and insurers are not, on this evidence, AI laggards. They are, however, slightly less loud about the fact than their software-industry peers are.

---

??? note "Evidence and sources"

    **Headline numbers**

    - Overall 2026 scraped SWE LinkedIn AI-strict prevalence = 13.24% (n=25,547; Wilson 95% CI 12.83-13.66%). `exploration/tables/journalist/industry_ai_prevalence.csv`; v1_rebuilt pattern 0.96 precision; labeled rows only. Fact-check 04 verified.
    - Financial Services 15.74% (n=1,785; CI 14.13-17.51%). Software Development 15.25% (n=6,870; CI 14.42-16.12%). Banking 14.52% (n=124), Insurance 15.00% (n=220), Hospitals & Health Care 20.87% (n=369). All fact-check-verified.
    - CIs for Financial Services and Software Development overlap substantially — the 0.49 pp difference is **not statistically significant**. Both sectors' CIs are non-overlapping with the 13.24% baseline CI (both are significantly above baseline).
    - BlackRock 44.4%, Wells Fargo 35.7%, SoFi 33.3% among firms with n ≥ 10. `exploration/tables/journalist/industry_ai_top_companies.csv`.
    - Named exemplars: Citigroup Gen AI-ML Engineer AVP (20 AI-pattern hits, 5.91 / 1K characters), JPMorgan SE III, Visa Staff SWE AI Agent, GEICO senior applied-AI, Wells Fargo fullstack, Goldman Sachs, BlackRock Director. `exploration/tables/journalist/finance_exemplars.csv`.
    - Applied-AI senior cluster (T34 cluster 0): 45% Software Development + 17% Financial Services in top-10 industry rows. Within-period only; LinkedIn industry taxonomy drifts between 2024 and 2026.

    **Conventional-wisdom opponent**

    Deloitte 2026 Banking & Capital Markets Outlook; McKinsey "Banking's AI Angst" (2025); Gartner 2025 AI Maturity Curve; Deskpro 2025 AI adoption report (92% vs 58%); *American Banker* opinion column on bank AI governance.

    **Sensitivity verdict**

    Industry cuts are within-2026-only (LinkedIn taxonomy drift forbids cross-period industry claims). Financial Services vs Software Development comparison: 0.49 pp gap with overlapping CIs — the piece claims parity, not advantage. Both sectors significantly above 13.24% baseline (non-overlapping CIs with baseline). Health care at 20.87% is a genuine surprise in the data and is addressed in the body. Pattern used: v1_rebuilt ai_strict at 0.96 precision. The piece claims finance *writes* dense AI postings, not that finance *operates* dense AI production systems — the consultancy narrative remains plausible on the latter.

---

## Related in Findings

- [Two new senior archetypes appear in 2026: "applied AI" and "forward-deployed"](../findings/a5-archetypes.md) — the Applied-AI / LLM Engineer archetype's 17% Financial Services share is quantified there.
