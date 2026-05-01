---
title: "Corners of the Screen — Table of Contents"
collection_status: in_progress
---

# Corners of the Screen
### *What the software-engineering job-postings data says when no one is listening.*

A collection of short Economist-style pieces anchored in a 68,000-posting longitudinal dataset of 2024-to-2026 LinkedIn software-engineering job advertisements. Each piece audits a widely-repeated claim about AI and software labour and reports the verdict.

---

## 00. [Introduction](00_introduction.md)
*The public argument, three-and-a-half claims contradicted, and the discipline behind the evidence.*

## 01. [The narrowing middle that wasn't](01_narrowing_middle.md)
*Two years of AI have reshaped software hiring. The junior rung is not where it thinned.* The junior share of labelled postings rose 5 percentage points from a 9.2% base — a roughly 55% relative rise — while five of six senior definitions fell. Seniority boundaries sharpened rather than blurred. The commonly cited Amodei and Brynjolfsson forecasts measure payroll; this measures live requisition composition, and the two can coexist.

## 02. [The Copilot paradox](02_copilot_paradox.md)
*The most-used AI tool in software engineering is nearly invisible in software-engineering job descriptions.* The word "copilot" in any form appears in 3.8% of 2026 postings; the bigram "GitHub Copilot" in roughly 0.1%. Stack Overflow's developer survey puts AI-assisted coding at 82% of working programmers. The gap is a factor of 21 under the broader pattern, a factor of roughly 400 under the narrower. Postings are an institutional record, with an institutional lag.

## 03. [Atlanta, not San Francisco](03_atlanta_not_san_francisco.md)
*The investment and talent remain in the Bay. The job descriptions that cite artificial intelligence are spreading everywhere else.* Across 26 American metros with meaningful SWE volume, the 2024-to-2026 rise in AI-requirement language is geographically flat — every metro gained, tech-hub premium below 2 percentage points, Tampa and Atlanta leading. The Bay Area's share of the new Senior Applied-AI archetype remains high, but the rate of change is elsewhere.

## 04. [What the compliance lobby can't see](04_finance_densest.md)
*Banks and insurers are supposed to be the AI laggards. Their job descriptions don't look it.* Pure-play Financial Services (15.7%) and pure-play Software Development (15.3%) are statistically indistinguishable on AI-requirement prevalence in 2026, both clearly above a 13.2% SWE baseline. Hospitals and health care, at 20.9%, exceed both. The "regulated-industry lag" narrative is priced by consultants, not by the people writing the postings.

## 05. [A decline that wasn't](05_management_never_there.md)
*A plausible-looking labour-market finding collapsed under independent scrutiny. Many plausible-looking findings in the same genre may be waiting for the same treatment, though nobody has performed it.* The first-pass "management density fell" finding was an artefact of an unvalidated regex pattern with 0.28 precision. The canonical posting-content papers — Deming & Kahn, Acemoglu et al., Lightcast — do not report precision validation. A reader cannot tell whether their findings would survive it.

## 06. [The rejection of the easy explanation](06_easy_explanations.md)
*Five plausible-sounding causes for the 2024-to-2026 rise in AI-requirement language. Four have been formally rejected; one has been partly reduced. The rise remains.* LLM-authorship contamination accounts for half of length growth and little of the content. Hiring-bar lowering, hiring-market selectivity, sampling-frame change and legacy-role rebranding all fail their own tests. What remains is the positive finding.

## 07. [The democratiser that asks for more experience](07_applied_ai_older.md)
*Large-language-model tooling is said to be compressing the experience premium in software. The single role most identified with the technology demands more experience, not less.* The Senior Applied-AI / LLM Engineer archetype grew 15.6× between 2024 and 2026. Its median YOE is 6.0 against its neighbours' 5.0, its director share nearly double. Access to programming may be democratising; the hiring bar for AI work is not.

## 08. [The market got the pattern right](08_ranking_right.md)
*Employers are said to be panicking about artificial intelligence, or oblivious to it, or both. They are doing something harder.* Across sixteen occupations — from software engineer to accountant to nurse — Spearman rank correlation between worker-side AI-usage and employer-side AI-requirement codification is +0.92. The level gap is enormous. The pattern is not.

## 09. [Forward-deployed, finally](09_forward_deployed.md)
*The most unexpectedly successful software-engineering title of the AI era was invented at a data-mining firm a decade ago. It has now escaped into defence, finance and professional services.* The Forward-Deployed Engineer archetype grew from 3 postings in 2024 to 59 in 2026 — a share rise of roughly seventeen-fold, across 38 distinct firms including Saronic Technologies, Govini, Mach Industries, CACI, Ramp, TRM Labs and PwC. At 2.3× the AI-density of the general SWE pool, it is the clearest single-title archetype in the corpus.

## 10. [Where the middle narrowed](10_where_middle_narrowed.md)
*The received wisdom blamed entry-level hiring. A revisionist view blamed the mid-career rung. Both looked in the wrong place.* Among five years-of-experience buckets tested, the 8-to-10-year rung is the only one that shrinks under both pooled and arshkon-only 2024 baselines (−3.3 pp and −1.0 pp respectively), intensifying to −3.7 pp on the 2,109-firm returning cohort. Junior and mid rose; very-senior fell. The ladder is not flatter, as commentators have argued — it is shorter at the top.

---

## Methodology notes
- **Dataset:** 68,137 SWE LinkedIn rows; arshkon 4,691 + asaniczka 18,129 + scraped 45,317; 2024 and 2026 snapshots.
- **Patterns:** V1-validated regex patterns at 0.92-1.00 semantic precision (`exploration/artifacts/shared/validated_mgmt_patterns.json`).
- **Seniority:** T30 multi-operationalisation panel; J3 (YOE≤2) and S4 (YOE≥5) primaries.
- **Verification:** V1 (Gate 2) and V2 (Gate 3) adversarial reviews. Every piece in this collection has been re-fact-checked by an independent Claude Opus 4.7 agent against its headline number, and independently critiqued by a second agent against its full draft. Revision notes are in each piece's evidence block.
- **Delegation trail:** investigator briefs and returns in `exploration/stories/_logs/`.

## Opponents explicitly named
- Dario Amodei (Axios, May 2025)
- Brynjolfsson et al., "Canaries in the Coal Mine?", Stanford Digital Economy Lab (August 2025)
- SignalFire State of Tech Talent (May 2025)
- Satya Nadella, quarterly earnings keynotes (2024-2026)
- Stack Overflow Developer Survey (2024)
- McKinsey "Superagency" (January 2025); BCG "AI at Work" (October 2025); WEF AI Perception Gap (January 2026); MIT/Fortune "Shadow AI Economy" (August 2025)
- Brookings "Mapping the AI Economy" (Muro et al., July 2025)
- Deloitte 2026 Banking Outlook; McKinsey "Banking's AI Angst"; Gartner 2025 AI Maturity
- Andrej Karpathy (Twitter, January 2023; "Software 3.0", 2025); Jensen Huang (February 2024); Sam Altman (March 2024); Matt Welsh (CACM, December 2022)
- Deming & Kahn 2018 JOLE; Acemoglu, Autor, Hazell & Restrepo 2022; Lightcast Global AI Skills Outlook 2024-2025
