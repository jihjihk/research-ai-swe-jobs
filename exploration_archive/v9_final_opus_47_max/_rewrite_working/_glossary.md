# Glossary — V9 rewrite

Canonical plain-English phrasings. Use these on first use of each term per page; duplication across pages is intentional. These are starting points — adapt to context.

## Data sources

- **arshkon** — "the 2024 arshkon dataset (a public Kaggle release of ~5,000 LinkedIn postings from April 2024; has native seniority labels, small sample)".
- **asaniczka** — "the 2024 asaniczka dataset (a second Kaggle release, ~18,000 LinkedIn postings from January 2024; four times larger than arshkon, no native seniority labels)".
- **scraped** — "our 2026 collection (~45,000 LinkedIn and Indeed postings scraped between March and April 2026)".

LinkedIn is the primary analysis platform; Indeed is used only for sensitivity.

## Samples — rename in prose

| Methodology page | Prose elsewhere |
|---|---|
| Sample 1 | the full software-engineering frame (68,000-posting LinkedIn corpus) |
| Sample 2 | the returning-firms cohort (2,109 firms posting in both 2024 and 2026) |
| Sample 3 | the within-firm panels (three panels of firms posting in both periods) |
| Sample 4 | the same-title pair panel (2024/2026 pairs from the same firm with the same title) |
| Sample 5 | the LLM-labeled subset (postings with a confident seniority classifier label) |
| Sample 6 | the archetype-labeled subset |
| Sample 7 | the interview exemplar set |

## Seniority codes — expand on first use

| Code | First-use expansion |
|---|---|
| J1 | J1 (labeled 'entry' by LinkedIn's native seniority field) |
| J2 | J2 (labeled 'entry' or 'associate' natively) |
| **J3** | J3 (postings asking for two or fewer years of experience — the study's primary junior definition) |
| J4 | J4 (rule-based YOE floor of two or fewer) |
| J5 | J5 (title keywords: intern/junior/jr/new grad) |
| S1 | S1 (labeled 'mid-senior' natively) |
| S2 | S2 (labeled 'director'; <1% of corpus, diagnostic only) |
| S3 | S3 (labeled 'mid-senior' or 'director') |
| **S4** | S4 (postings asking for five or more years of experience — the study's primary senior definition) |
| S5 | S5 (rule-based YOE floor of five or more) |

"Junior" and "senior" in plain English beat the codes when the specific definition doesn't matter.

## Finding codes — never as headline

| Code | Plain-English headline |
|---|---|
| A1 | Employers describe AI work at a fraction of the rate workers report using it |
| A2 | When firms add AI language to postings, they rewrite the same titles rather than invent new ones |
| A3 | Junior and senior job descriptions moved apart between 2024 and 2026, not together |
| A4 | Every tier asks for broader skills than in 2024; senior roles gained more breadth than junior |
| A5 | Two new senior archetypes appear in 2026: "applied AI" and "forward-deployed" |
| A6 | The software-engineering vs. control gap doesn't depend on which controls we picked |

Cross-reference label ("see A3 for the seniority analysis") is fine after the plain-English framing has been used once.

## Codes to strip from prose

Keep in audit-trail links and footers. Do not write in body prose.

- **T01–T38** → the thing the task measured.
- **V1, V2, V2 Phase E** → "a follow-up replication" / "a robustness check".
- **Wave N, Gate N** → strip entirely.
- **H_A–H_Q** → name the hypothesis in English.
- **RQ1–RQ4** → name the question in English on first use.

## Statistical terms — gloss on first use per page

- **Difference-in-differences (DiD):** "a difference-in-differences comparison (the 2024-to-2026 change in SWE postings minus the change in control occupations, so a labour-market-wide shock would cancel out)". "DiD" alone after first use.
- **Percentage points (pp):** "14 percentage points (the difference between 10% and 24%, not a 14% proportional rise from 10%)".
- **Area under the curve (AUC):** "an area-under-curve score (how well a classifier separates two groups; 0.5 is chance, 1.0 is perfect)".
- **Cosine similarity:** "a cosine similarity score (a 0-to-1 measure of shared vocabulary between two documents; closer to 1 means more overlap)".
- **TF-IDF:** "TF-IDF weighting (emphasises distinctive words, downweights common filler)".
- **Spearman correlation (ρ):** "a Spearman rank-correlation (+1 means two measures order items identically; 0 means no relationship)".
- **Wilson confidence interval:** "95% Wilson CIs (a standard interval estimate for proportions; overlapping intervals mean the difference isn't statistically distinguishable at the 95% level)".
- **Bootstrap CI:** "a bootstrap 95% CI (a resampling estimate of how much the statistic could vary)".
- **Signal-to-noise ratio (SNR):** "a signal-to-noise ratio of 1.06 (the 2024-to-2026 move is roughly the same size as the within-2024 variation — near-noise, read as directional only)".
- **Herfindahl index (HHI):** "a Herfindahl index of 0.14 (concentration measure: 0 is evenly distributed, 1 is single-firm dominance)".
- **Silhouette score:** "a silhouette of 0.477 (cluster-quality score from −1 to 1; ~0.5 is mid-range, not clean separation)".
- **BERTopic:** "BERTopic (a topic-modelling library that groups documents by meaning rather than exact-word overlap)".

## Research-concept terms

- **Worker-side benchmark / employer-side rate:** "we compare two numbers per occupation: how often *workers* report using AI on the job (survey data), and how often *postings* mention AI. The first is the worker-side benchmark; the second is the employer-side rate".
- **Returning-firms cohort:** "the 2,109 firms that posted software-engineering roles in both 2024 and 2026".
- **Same-title pair panel:** "pairs of postings from the same firm for the same job title, one from 2024 and one from 2026".
- **Pattern-based AI detection (was `ai_strict`):** "our pattern-based AI-mention detector (catches named AI tools like Copilot and Cursor, LLM-related terms, and AI-coding-assistant mentions)". Don't write `ai_strict` in prose.
- **Scope inflation / breadth score:** "the number of distinct skill or tool requirements per posting, residualised for posting length".
- **LLM-labeled subset:** "postings where our language-model seniority classifier produced a confident label". Avoid "LLM-frame".
- **Applied-AI / forward-deployed archetypes:** names are content-grounded; gloss briefly on first use — "'applied AI' (senior engineers working with production LLM systems)" and "'forward-deployed' (customer-facing senior engineers integrating AI at client sites)".

## When in doubt

- Technical and important on this page → explain it.
- Technical but passing → replace with plain English; skip the gloss.
- Internal code with no external meaning → strip; keep the link in the footer.
- Plain-English phrasing genuinely awkward → use the code once, gloss it, move on.
