---
title: New angles from HR / recruiting industry discourse
scope: hr-recruiting
claims_count: 9
date: 2026-04-21
---

# New angles — HR / recruiting / talent-analytics

Sources: Lightcast, Revelio, Indeed Hiring Lab, Korn Ferry, Dice, Ashby, ResumeUp, Greenhouse, Burning Glass Institute, SHRM. Nine claims testable against our 68k-SWE LinkedIn scrape, orthogonal to or contrarian against the 8 drafted pieces.

---

### HIGH 1 — Lightcast: AI skills +28% pay; 51% of AI postings OUTSIDE IT

- **Src:** Lightcast *Beyond the Buzz* (Jul 2025). https://lightcast.io/resources/blog/beyond-the-buzz-press-release-2025-07-23
- **Claim:** AI-skill postings pay 28% more (~$18k/yr); 51% outside IT/CS as of 2024 (1.3B postings).
- **Data-led.**
- **Test:** Regress log(salary_max) on AI-skill flag within SWE. Separately: AI-skill share by industry.
- **Vs 8:** Orthogonal. #2 (Copilot) and #7 (Applied-AI YOE) never test a within-SWE AI premium.
- **Headline:** "Within software, the AI pay premium is a rounding error" — fires if <10%.

### HIGH 2 — Korn Ferry: AI premium only 5-15%; $200k cliff

- **Src:** Korn Ferry *Only Modest Pay Bumps for AI Skills* (2025). https://www.kornferry.com/insights/this-week-in-leadership/only-modest-pay-bumps-for-ai-skills
- **Claim:** 5-15% premium only. Firms offering <$200k for senior AI roles: **114-day** time-to-fill vs **52 days** market-wide.
- **Data-led.**
- **Test:** Posting lifetime (first-seen → last-seen) as proxy. Bin by salary_max <$200k vs ≥$200k × AI × senior.
- **Vs 8:** Directly contradicts #1's 28%. None of 8 have a compensation-cliff angle.
- **Headline:** "The $200k line: below it, senior AI postings linger twice as long."

### HIGH 3 — Indeed: SWE postings -49% vs Feb 2020; Android/iOS/.NET each -60%+

- **Src:** Indeed Hiring Lab *US Tech Hiring Freeze Continues* + *Experience Requirements Have Tightened* (Jul 30 2025). https://www.hiringlab.org/2025/07/30/the-us-tech-hiring-freeze-continues/
- **Claim:** SWE -49%; Android/Java/.NET/iOS/web -60%+; experience requirements tightened in the shrinking pool.
- **Data-led.**
- **Test:** Mar 2024 → Mar 2026 posting trajectory by stack token (java, android, ios, web, .net) vs generic SWE vs AI/ML. Track mean YOE.
- **Vs 8:** Complements #1 (narrowing middle) but pivots to **stack** as loser. None of 8 look at stack decay.
- **Headline:** "Android, iOS, .NET: three stacks that fell off a cliff while LinkedIn kept posting."

### HIGH 4 — Ghost-job epidemic: 27.4% on LinkedIn; 40% of tech firms admit

- **Src:** ResumeUp.AI via Entrepreneur (2025) https://www.entrepreneur.com/business-news/one-quarter-of-jobs-posted-online-are-fake-ghost-jobs-study/496683 ; CNBC Nov 11 2025 https://www.cnbc.com/2025/11/11/ghost-job-postings-add-another-layer-of-uncertainty-to-stalled-jobs-picture.html ; Greenhouse 2025 CX: 36% applied to a never-filled role.
- **Claim:** 27.4% US LinkedIn listings are ghost jobs; 40% of tech firms admit; 80%+ of recruiters have done it; 41% say >half.
- **Data-led.**
- **Test:** Reposting rate, duration distribution (alive >60/90/120d), duplicate-description ratio, disappear-reappear cycles. Split by seniority and company.
- **Vs 8:** Brand-new axis. #1 (narrowing middle) implicitly assumes postings = real demand.
- **Headline:** "The missing juniors: when a third of 'entry-level' postings were never meant to be filled."

### HIGH 5 — Ashby: engineering time-to-hire = 85 days, 52% offer acceptance

- **Src:** Ashby Talent Trends Reports 2024/25. https://www.ashbyhq.com/talent-trends-report/reports/startup-hiring
- **Claim:** Engineering = worst funnel of any function; ~5.4 SWE hires/recruiter/quarter.
- **Data-led (ATS panel).**
- **Test:** Posting lifetime proxy. Share still-alive at day 85 by seniority, AI-vs-not, industry (finance vs tech).
- **Vs 8:** Orthogonal. #4 (finance densest) measures density not churn.
- **Headline:** "The 85-day posting: if engineering hiring takes three months, LinkedIn shows it."

### MEDIUM 6 — Dice: AI-involved +17.7% pay — partly seniority-mix artifact

- **Src:** Dice *2025 Tech Salary Report* (Feb 2025). https://www.dice.com/career-advice/dice-2025-tech-salary-report-which-tech-skills-pay-you-the-most
- **Claim:** AI-involved pros earn 17.7% more; Dice attributes part of gap to AI skewing CEO/CTO/Director.
- **Data-led.**
- **Test:** Decompose AI premium into (a) level-mix vs (b) within-level via title fixed effects / Oaxaca.
- **Vs 8:** Reinforces #7 (Applied-AI YOE).
- **Headline:** "The AI pay premium is a seniority illusion" — if level-mix explains >70%.

### MEDIUM 7 — Revelio: AI-exposed task share fell 29% → 25.5% (2022 → 2025)

- **Src:** Revelio Labs *Tasks You Won't See in Job Postings Anymore* (2025). https://www.reveliolabs.com/news/macro/the-tasks-you-won-t-see-in-job-postings-anymore/
- **Claim:** AI-exposed tasks lost 3.5pp of JD share; data/doc clerks lost 8.2pp.
- **Data-led (longitudinal).**
- **Test:** SWE-specific AI-exposed task taxonomy (boilerplate coding, unit tests, code review, docs); track share Mar 2024 → Mar 2026.
- **Vs 8:** Orthogonal — task- not skill-level. Reframes #2 (Copilot paradox) from tool adoption to task disappearance.
- **Headline:** "Quietly, 'write unit tests' disappeared from SWE JDs."

### MEDIUM 8 — Indeed + BGI: Bachelor's-required rebounded to 19.3%; skills-based hiring is a press release

- **Src:** Indeed Hiring Lab *Where Do College Degrees Still Matter* (Jan 28 2026) https://www.hiringlab.org/2026/01/28/where-do-college-degrees-still-matter-in-a-skills-first-job-market/ ; Burning Glass Institute *Skills-Based Hiring 2024* https://www.burningglassinstitute.org/research/skills-based-hiring-2024
- **Claim:** Bachelor's-required share rose to 19.3% Nov 2025 (from 17.8% Jan 2024); 85% claim skills-based hiring, <0.15% of hires actually affected.
- **Data-led.**
- **Test:** Share mentioning BS/BA/Bachelor's required vs preferred vs absent, monthly, by industry.
- **Vs 8:** Orthogonal to all 8.
- **Headline:** "Skills-based hiring is a press release. LinkedIn job ads still want the degree."

### MEDIUM 9 — Returnships: IBM, LinkedIn, Goldman, JPM, Cisco all run SWE tracks

- **Src:** SHRM *Returnships* https://www.shrm.org/topics-tools/tools/express-requests/returnships-workforce-re-entry-programs ; Techneeds *10 SWE Returnship Programs* (Oct 2025) https://www.techneeds.com/2025/10/20/10-software-engineer-returnship-programs-to-boost-your-hiring/
- **Claim:** Major employers advertise SWE re-entry programs; industry coverage expanded 2024-25.
- **Forecast/qualitative.**
- **Test:** Grep "returnship," "re-entry," "career break," "return to work." Count by employer/industry, cross-tab vs AI-skill.
- **Vs 8:** Ties to #7 (Applied-AI YOE older). If returnships cluster in finance/defense not AI → contrast piece.
- **Headline:** "The returnship map: where employers recruit the 35+ SWE that AI firms won't."

---

**Triage.** Pursue first: #1, #2, #4, #5. Corroborative: #3, #6. Exploratory: #7, #8, #9.
