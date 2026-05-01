---
title: New angles from policy / government / public-interest discourse
scope: policy
claims_count: 8
date: 2026-04-21
---

# New angles — policy and government research

Scan of BLS, Fed Board + regional banks, Stanford DEL, OPM/OSTP, Brookings, USCIS/EPI, GAO. 8 testable claims.

---

## HIGH — 1. Stanford DEL "software devs 22-25 fell ~20% from late-2022 peak"

- **Source:** Brynjolfsson, Chandar & Chen, *Canaries in the Coal Mine?*, Stanford Digital Economy Lab, Aug/Nov 2025. https://digitaleconomy.stanford.edu/publication/canaries-in-the-coal-mine-six-facts-about-the-recent-employment-effects-of-artificial-intelligence/
- **Claim:** SWE 22-25 fell ~20% from late-2022 peak; 16% relative decline net of firm shocks. Decline occurred only in *automative* AI applications, not augmentative.
- **Type:** Data-led (ADP payroll).
- **Test:** Split postings into automation-exposed vs augmentation-exposed AI tokens. Volume shrinking only for automate-type roles = clean replication beyond #1.
- **Vs our 8:** Agrees with #1, adds a mechanism split we lack.
- **Contrarian headline:** "Stanford's canaries replicate — but only on the automating side."

## HIGH — 2. Fed FEDS Note "no evidence AI adoption reduced job postings"

- **Source:** Liu & Webber, *AI Adoption and Firms' Job-Posting Behavior*, FEDS Notes, 27 Mar 2026. https://www.federalreserve.gov/econres/notes/feds-notes/ai-adoption-and-firms-job-posting-behavior-20260327.html
- **Claim:** Lightcast+BTOS (Sep 2023-Nov 2025): firm-level "plan to use AI" coefficient +0.082*** at 1-month lag, ~0 at 12-month. Higher-AI industries do NOT post fewer jobs; effect +0.04-0.13%.
- **Type:** Panel regression.
- **Test:** Bin our panel by firm AI-adoption proxy (count of prior AI-skill postings per employer). Windows overlap almost exactly.
- **Vs our 8:** Contrarian vs #1 if high-adopter firms expand SWE.
- **Contrarian headline:** "The Fed says AI firms aren't posting fewer jobs. Our SWE panel agrees — with one exception."

## HIGH — 3. St. Louis Fed "0.57 correlation: more AI adoption = more unemployment"

- **Source:** St. Louis Fed, *Is AI Contributing to Rising Unemployment?*, Aug 2025. https://www.stlouisfed.org/on-the-economy/2025/aug/is-ai-contributing-unemployment-evidence-occupational-variation
- **Claim:** Corr 0.57 between 2022-25 unemployment change and AI adoption intensity; computer/math at ~80% exposure saw steepest rises.
- **Type:** Correlational (Fed disclaims causation).
- **Test:** Rank SWE sub-specialties (frontend/backend/ML/data-eng/SRE/mobile) by posting change 2024-26, correlate with AI-intensity scores. Does 0.57 hold *within* SWE?
- **Vs our 8:** Deepens #5 and #6.
- **Contrarian headline:** "The 0.57 that doesn't survive zooming into software."

## HIGH — 4. OPM "US Tech Force — 1,000 federal AI fellows, $130k-$200k, Q1 2026"

- **Source:** OPM/OSTP/OMB, *OPM Launches US Tech Force*, 15 Dec 2025. https://www.opm.gov/news/news-releases/opm-launches-us-tech-force-to-implement-president-trumps-vision-for-technology-leadership/
- **Claim:** 1,000 Fellows/yr at $130-200k for AI/cyber/SWE; 250 Data Science Fellows Spring 2026. Partners include OpenAI, Palantir, Anduril, xAI, Databricks.
- **Type:** Policy commitment.
- **Test:** Look for Q1 2026 uptick in federal/defense-contractor SWE postings (Booz Allen, Leidos, SAIC, Anduril, Palantir, .gov).
- **Vs our 8:** Fully orthogonal.
- **Contrarian headline:** "The federal AI hiring surge is crowding out private SWE openings in DC, not Silicon Valley."

## MEDIUM — 5. BLS 2024-34 "SWE +15% = 129k annual openings, no AI displacement priced in"

- **Source:** BLS, *Employment Projections 2024-2034*. https://www.bls.gov/news.release/pdf/ecopro.pdf ; https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm
- **Claim:** SWE+QA+testers +15% 2024-34 (from 1.7M); 129,200 avg annual openings. BLS cites AI/IoT/cyber as *demand drivers*, does NOT price in AI displacement.
- **Type:** Official projection.
- **Test:** Compare 2024→2026 monthly posting trend to the implied ~+1.5%/yr BLS trajectory.
- **Vs our 8:** Reference baseline.
- **Contrarian headline:** "BLS projected 129k new SWE openings/year. Through 2026, LinkedIn runs X% short."

## MEDIUM — 6. Brookings "Bay Area = 13% of AI postings; top 30 metros = 67%"

- **Source:** Muro et al., *Mapping the AI Economy*, Brookings, 2025. https://www.brookings.edu/articles/mapping-the-ai-economy-which-regions-are-ready-for-the-next-technology-leap/ ; *Building AI Cities*, 2025. https://www.brookings.edu/articles/building-ai-cities-how-to-spread-the-benefits-of-an-emerging-technology-across-more-of-america/
- **Claim:** Bay Area = 13% of AI-tagged postings; top 30 metros = 67%. Enterprise AI adoption: 4% (early 2023) → 8.7% (mid-2025).
- **Type:** Data-led (Lightcast).
- **Test:** Bay Area share of SWE vs AI-SWE postings over time. Falling Bay share + rising AI share = dispersion story.
- **Vs our 8:** Adjacent to #3; gives benchmarks.
- **Contrarian headline:** "Brookings says 67% of AI jobs sit in 30 cities. For SWE, the top 30 holds only X%."

## MEDIUM — 7. DHS/USCIS "H-1B goes wage-weighted Feb 2026 — SWE median LCA $149k"

- **Source:** DHS/USCIS final rule, 29 Dec 2025, effective 27 Feb 2026. https://public-inspection.federalregister.gov/2025-23853.pdf ; EPI context: https://www.epi.org/publication/h-1b-visas-and-prevailing-wage-levels/
- **Claim:** Wage-weighted, beneficiary-centric H-1B lottery replaces random; FY26 avg SWE LCA $149,407 (30,014 filings).
- **Type:** Policy + admin data.
- **Test:** Compare 2024 vs 2025 posted-salary distributions at H-1B-heavy employers (Cognizant, Infosys, Amazon, Google, Meta, TCS) vs non-sponsors. Lower-tail lift = anticipatory behavior.
- **Vs our 8:** Orthogonal — no piece covers immigration.
- **Contrarian headline:** "Big Tech repriced low-tier SWE postings months before the H-1B rule took effect."

## MEDIUM — 8. GAO/ClearanceJobs "70k unfilled cleared roles; 63,934 federal cyber staff"

- **Source:** GAO-25-106795 & GAO-25-107405, 2025. https://www.gao.gov/products/gao-25-106795 ; https://www.gao.gov/assets/gao-25-107405.pdf
- **Claim:** 63,934 federal + 4,151 contractor cyber roles (Apr 2024). 56% of recruiters cite cleared talent as #1 challenge; ~70k cleared-role vacancies.
- **Type:** Agency self-report + survey.
- **Test:** Flag clearance language ("Secret/TS/SCI/Polygraph") in SWE postings; track share and posting-duration gap vs uncleared.
- **Vs our 8:** Fully orthogonal.
- **Contrarian headline:** "The slowest-filling software jobs in America need a security clearance — not an AI skill."

---

## Flags

- **Top replication priority:** #1 Stanford canaries — most cited, most directly testable on our YOE data.
- **Best new-piece candidates:** #4 Tech Force, #7 H-1B rule, #8 cleared workforce — none overlap.
- **Best "sharpen existing":** #6 Brookings upgrades geography piece #3 with a concrete benchmark.
- **Skipped:** CBO +10bps macro (too aggregate); IMF 60% exposure (redundant with Stanford + FEDS); OECD management-skills (covered by Applied-AI YOE); DOL Best Practices (not quantitative); CHIPS Act (not core SWE).
