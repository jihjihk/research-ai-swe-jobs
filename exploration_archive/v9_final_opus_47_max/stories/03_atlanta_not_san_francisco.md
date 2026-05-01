---
title: "Atlanta, not San Francisco"
slug: 03_atlanta_not_san_francisco
word_count_target: 500-650
status: revised_after_factcheck_critic
---

# Atlanta, not San Francisco

*The investment and the talent remain in the Bay. The job descriptions that cite artificial intelligence are spreading everywhere else.*

The geography of American artificial intelligence, as the mainstream press reports it, is a geography of agglomeration. PitchBook's fourth-quarter 2025 venture-capital digest finds that the Bay Area absorbs 52% of American AI/ML investment dollars. CBRE's 2025 tech-talent report records the Bay Area's stock of AI-skilled workers growing at six times the national rate. Brookings' July 2025 "Mapping the AI Economy" report, which builds on job-posting data, concludes that San Francisco and San Jose are "unmatched superstars" in AI employment. Investment, talent stock, and even AI-specific job-posting counts all concentrate in two zip codes.

The *rate of change* in AI-requirement language across software-engineering job descriptions is a different story.

Among 26 American metropolitan markets with at least 50 software-engineering postings in both 2024 and 2026, the 2024-to-2026 rise in the share of postings that cite AI tools or skills is geographically uniform. Every metro recorded a positive gain; under an independent re-derivation on the curated core dataset, gains ranged from roughly +6 to +19 percentage points across metros, with a tech-hub premium — the extra gain in San Francisco, Seattle, New York, Austin and Boston relative to the rest — of 1.65 points. The uniformity is independent of aggregator inclusion and of company posting-volume caps.

The four metros that rose the most on AI mentions are Tampa, Atlanta, Minneapolis and Charlotte, on our independent recount; the task report from which this investigation began names Tampa, Atlanta, Miami and Salt Lake City — the top two, Tampa and Atlanta, replicate cleanly. The Bay Area's AI-mention gain is in the middle of the distribution, above the national average by about a point. Seattle's is slightly below. On the specific share of postings that match the ML/LLM archetype, Minneapolis gained more ground than Seattle — +13 points against +8.8.

There is one thing the Bay Area does continue to lead on, and it is the absolute count, not the rate. Roughly 26% of postings for the newly emergent "Senior Applied-AI / LLM Engineer" archetype are listed in the Bay Area; Dallas–Fort Worth and Seattle follow. The AI role that most resembles what OpenAI or Anthropic do internally remains hub-weighted. What is not hub-weighted is the writing of AI requirements into ordinary software-engineering postings. A bank in Charlotte or an insurer in Tampa can decide to add Claude, RAG and LangChain to a Java-backend requisition without waiting for a local cluster. Nineteen percent of the 2026 corpus is listed as remote, and the remote AI-mention rate (11.1%) is essentially identical to the metro average (10.5%). Where the work is described, for ordinary engineering hires, is decoupled from where the money lives.

The investment concentration and the posting-rate uniformity measure different things, and the piece's claim is narrow: for the content of job descriptions in software-engineering postings between 2024 and 2026, metro-level differences are small. The junior-share change, by contrast, moves quite differently across metros, from roughly −5 points in Detroit to +15 in San Diego; the correlation between a metro's AI-requirement change and its junior-share change across the 26 markets is effectively zero (Pearson r = −0.22 at n=26, not statistically distinguishable from no correlation). Two phenomena that mainstream coverage often treats as a single narrative — "AI is eating the junior rung, and AI is concentrating in hubs" — turn out, in the posting content, to be uncorrelated.

The geography of posting-content AI is not a geography of diffusion (that would require a longer time series); it is a geography of flatness. For a journalist looking to explain which American cities are adapting their ordinary software-engineering job descriptions around AI, the Bay Area is the wrong place to look. It is a place that already had.

---

## Evidence block (not for publication)

1. **AI-rise range across 26 metros on independent `unified_core` re-derivation: +6.45 to +18.60 pp, all positive; matches T17's reported +4.7 to +14.5 pp qualitatively (different n's on different file frames).** Fact-check 03.
2. **Tech-hub premium (SF + NYC + Seattle + Austin + Boston mean vs rest): +1.65 pp on unified_core; +1.3 pp per T17.** Fact-check 03.
3. **Top 4 AI-rise metros (unified_core): Tampa, Atlanta, Minneapolis, Charlotte. T17 top 4: Tampa, Atlanta, Miami, SLC. Tampa + Atlanta replicate at top.** Fact-check 03.
4. **Minneapolis +13 pp vs Seattle +8.8 pp on ML/LLM archetype share.** T17 + T09.
5. **r(metro AI-rise, metro J3-rise) = −0.22 at n=26 (not statistically significant; p ≈ 0.28).** T17.
6. **Remote share 19.9% of 2026; remote AI rate 11.1% vs metro avg 10.5%.** T17.
7. **Applied-AI senior archetype geography: SF 26%, DFW 11.4%, Seattle 11.1%.** T34.

**Conventional-wisdom opponent:** PitchBook Q4 2025 (AI investment — 52% of US VC in Bay Area); CBRE 2025 tech-talent report (AI-skilled worker stock); Brookings "Mapping the AI Economy" Muro et al. (July 2025; uses posting data to argue SF/San Jose are "unmatched superstars"). Piece distinguishes investment dollars, talent stock, and posting-content-rate — three different constructs.

**Sensitivity verdict:** The 26-metro panel is restricted to markets with ≥50 SWE postings per period. Metro-level CI on any individual metro's AI delta is wide; the uniformity claim is about the distribution of deltas, not any one metro. Tech-hub premium is sub-2 pp on both T17 and fact-check 03 recount. r = −0.22 at n=26 is not statistically significant (p ≈ 0.28); the piece now reports this honestly rather than claiming decoupling. "Diffusion" framing has been removed; replaced with "flatness," which is what the cross-sectional data actually shows (without a pre-2024 panel, true diffusion cannot be tested).

**Revision note:** Separated three conflated sources (investment, talent stock, posting rates) at the opening. Replaced "diffusion" language with "flatness" (flat cross-section is what the data shows; diffusion would require time-series). Honest framing of r = −0.22 as "not statistically distinguishable from no correlation." Surfaced tech-hub premium number in body. Closing softened from "started and moved on from" to "already had" — still punchy, no longer implying an unobserved trajectory.
