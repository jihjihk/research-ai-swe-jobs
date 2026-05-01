---
title: "Prior-art: regex/keyword methods on JD text without precision validation"
sources: 4
investigator: "prior-art"
piece: "05_management_never_there"
---

# Prior art: unvalidated regex/keyword methods on job-posting text

The methodological practice under scrutiny — measuring a latent construct (management, social skills, AI skills) with a regex or keyword list applied to JD text, then tracking it over time — is not an edge case. It is the dominant template of the posting-content literature. Below are four recent, heavily-cited studies that (a) use keyword/regex/taxonomy tagging on a posting corpus to measure a latent construct and (b) do not report semantic precision validation of the patterns used. The last entry is the counter-example: the only prominent study in this family that explicitly validated its dictionary and then rejected it.

## 1. Deming & Kahn (2018) — "Skill Requirements across Firms and Labor Markets"

- Authors: David Deming (Harvard), Lisa B. Kahn (Rochester)
- Outlet: Journal of Labor Economics 36(S1). NBER WP 23328.
- URL: https://www.nber.org/papers/w23328
- Method: Categorize ~45M Burning Glass postings (2010–2015) into 10 skill buckets (cognitive, social, character, writing, customer-service, project-management, people-management, financial, general-computer, specific-computer) using keyword/phrase lists applied to posting text.
- Validation: No. The paper defines each bucket with a keyword table and interprets the prevalence and covariance with wages/firm outcomes directly. Precision of the keyword matches (i.e., "what share of the ads flagged as 'people-management' actually describe management responsibilities?") is not reported. This paper is the most-cited template for subsequent dictionary-based JD studies.

## 2. Acemoglu, Autor, Hazell & Restrepo (2022) — "AI and Jobs: Evidence from Online Vacancies"

- Outlet: Journal of Labor Economics 40(S1). NBER WP 28257.
- URL: https://www.nber.org/papers/w28257
- Method: Flag Burning Glass vacancies as "AI-exposed" using a list of AI-related skill tags/keywords (machine learning, natural language processing, neural networks, etc.) and track establishment-level hiring changes over 2010–2018.
- Validation: No precision check on the AI-keyword list reported. The paper relies on the Burning Glass skill taxonomy; no human-coded precision sample, no check that flagged postings are genuinely AI-substitutive rather than, e.g., marketing boilerplate mentioning "AI-powered." Their causal design controls for establishment fixed effects but treats the keyword tag as ground truth.

## 3. Lightcast / Brookings — AI-skills outlook reports (2024–2025)

- Publishers: Lightcast (formerly Burning Glass Technologies); cited by Brookings (Muro et al.), CBS News, Atlanta Fed (Oct 2024; May 2025).
- URLs: https://lightcast.io/resources/research/the-lightcast-global-ai-skills-outlook ; https://www.brookings.edu/articles/new-data-show-no-ai-jobs-apocalypse-for-now/
- Method: "AI job posting" = any posting tagged with at least one skill from Lightcast's ~250-item AI skill list within their 32,000-skill Open Skills Taxonomy. Widely cited headline numbers (66k generative-AI postings in 2024, 80k in 2025, 28.5% annualized growth since 2010) flow from this tagging.
- Validation: No. Lightcast's public documentation describes the taxonomy's coverage and hierarchy but — per their own Outlook page — "does not disclose any precision validation, accuracy metrics, or testing results for their AI skill tagging methodology." Downstream Brookings write-ups inherit the tag and do not re-validate.

## 4. Hansen, Lambert, Bloom, Davis, Sadun & Taska (2023/2025) — "Remote Work across Jobs, Companies, and Space" (COUNTER-EXAMPLE)

- Outlet: NBER WP 31007 (updated 2025).
- URL: https://www.nber.org/papers/w31007
- Method: Fine-tune a language model on 30,000 human-labeled postings to classify WFH arrangements across 500M vacancies.
- Validation: Yes — and the result is the point. The authors explicitly benchmark their LLM (99% accuracy) against a dictionary approach and report the dictionary "substantially" underperforms, because "remote" is ambiguous in context ("this position allows remote work" vs. "this position involves working on remote sites"). The finding is stated in the paper and reiterated on the WFH Map project page. It is a rare, published demonstration that the default dictionary method on posting text can be wrong by double-digit percentage points for a single word — yet the literature above continues to use dictionary tagging elsewhere without analogous checks.

## Backdrop for the piece

Three of the four dominant templates in longitudinal JD-content work — Deming-Kahn's skill-bucket regex, Acemoglu-Autor-Hazell-Restrepo's AI-keyword tag, and the Lightcast taxonomy feeding Brookings/Atlanta-Fed commentary — do not report posting-level precision validation. The Hansen-Bloom-Davis paper is the exception that proves the rule: when someone did check, dictionary methods failed, and by enough that the authors built a whole LLM pipeline to replace them. Our 0.28-precision "management" regex sits squarely in the unvalidated-majority zone; its failure mode (a trend artifact driven by "contract-to-hire" and "code review" boilerplate) is exactly what Hansen et al. warned about for "remote."
