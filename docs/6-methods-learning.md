# Methods Learning Guide

This is a supporting learning document derived from the methodology items flagged in the March 6, 2026 meeting notes. It explains what each method is, what it is good for, where it can fail, and how it could fit this project.

It is not a canonical design document. Read it after:

1. `docs/1-research-design.md`
2. `docs/2-interview-design-mechanisms.md`
3. `docs/3-literature-review.md`
4. `docs/4-literature-sources.md`
5. `docs/5-publication-targets-2026-2027.md`

## Why This Exists

The March 6 notes surfaced a set of methods that are genuinely useful, but they play different roles:

- some are hypothesis-generation tools
- some are measurement tools
- some are qualitative-analysis methods

Treating them as interchangeable would weaken the study. The right move is to understand what each method contributes and then assign it a narrow, defensible role.

## 1. Topic Models

### What they are

Topic models are unsupervised methods that identify recurring patterns of word co-occurrence across documents.

The classical reference is **Latent Dirichlet Allocation (LDA)**, which models each document as a mixture of latent topics and each topic as a distribution over words. Later work such as the **Structural Topic Model (STM)** allows document-level covariates like time, seniority, or source to shape topic prevalence and topic content. More recent methods such as **BERTopic** replace bag-of-words document representations with embedding-based clustering and then summarize clusters using class-based TF-IDF.

### What they are good for

- surfacing broad themes without hand-labeling every document
- hypothesis generation
- discovering requirement bundles that may not be obvious from a keyword list
- comparing how themes shift across periods, seniority levels, or sources

### What they are bad for

- proving a precise substantive claim on their own
- replacing direct measurement of named constructs
- producing stable topics without researcher judgment
- handling very short or noisy documents gracefully

Topic models are exploratory. They are useful for seeing the landscape, but they are usually too unstable and too researcher-dependent to serve as the only evidence for a paper's headline claim.

### Which variant matters here

- `LDA`: useful baseline, interpretable, but often too coarse for modern job-posting text
- `STM`: stronger if you want topic prevalence by `year`, `junior/senior`, `LinkedIn/Indeed`, or `metro`
- `BERTopic`: often useful for modern corpora because job postings contain semantically similar phrases that embeddings handle better than raw bag-of-words alone

### How it could help our research

For this project, topic models should be used for **hypothesis generation and robustness**, not as the main evidence layer.

Good uses:

- identify emergent themes in 2026 junior postings that were rare in 2023-2024
- compare junior and senior topic prevalence over time
- inspect whether AI-tool language co-occurs with ownership, architecture, or systems language

Bad use:

- claiming "topic X proves junior scope inflation" without validating the topic manually

### Recommendation

Use BERTopic or STM on a deduped sample of cleaned descriptions after basic keyword analyses are already defined. Then manually inspect topic exemplars and treat topics as prompts for better dictionaries, not as the final measurement instrument.

## 2. The "Fightin' Words" Method

### What it is

**Fightin' Words** is a weighted lexical-comparison method from Monroe, Colaresi, and Quinn (2008). It identifies words or features that are disproportionately associated with one corpus relative to another, while using Bayesian shrinkage to avoid overreacting to rare terms.

This is much better than naive "top words by frequency difference" because it regularizes unstable estimates and highlights words that are meaningfully distinctive.

### What it is good for

- comparing two corpora directly
- finding distinctive language in junior vs senior postings
- finding distinctive language in 2023-2024 vs 2026 postings
- comparing LinkedIn vs Indeed language

### What it is bad for

- measuring latent concepts that need context, not just lexical contrast
- handling synonyms or paraphrases without preprocessing
- making claims about tasks if the same word means different things in different contexts

### Why it is attractive for this project

It gives a fast, interpretable answer to questions like:

- Which terms became much more distinctive in junior postings by 2026?
- Which terms distinguish senior postings now from senior postings in 2023-2024?
- Which words disproportionately appear in "entry-level" postings that look inflated?

That makes it one of the best early-stage tools for the corpus you are building.

### Recommendation

Use Fightin' Words early.

Suggested contrasts:

- junior 2026 vs junior 2023-2024
- junior 2026 vs senior 2023-2024
- senior 2026 vs senior 2023-2024
- LinkedIn vs Indeed within the same seniority tier

Then manually inspect the top lexical contrasts and fold the strongest patterns into dictionaries or annotation rules.

## 3. Inductive Thematic Analysis

### What it is

Thematic analysis identifies patterned meaning across qualitative data. The most relevant reference here is Braun and Clarke's work, especially their 2006 paper and later clarification on **reflexive thematic analysis**.

Inductive thematic analysis means codes and themes are generated from the data rather than imposed entirely in advance.

### What it is good for

- interview data
- open-ended practitioner explanations
- understanding mechanisms and meanings
- surfacing themes you did not pre-specify

### What it is bad for

- pretending to be purely mechanical or objective
- generating "themes" without interpretation
- replacing quantitative measurement where precise counts matter

### Important distinction

There are at least two families of thematic analysis relevant here:

- `reflexive thematic analysis`: interpretation-forward, suited to meaning and lived experience
- `codebook / coding-reliability thematic analysis`: more structured, better when teams want a shared coding frame and partial standardization

For this project, the interview study likely benefits more from **reflexive thematic analysis** than from a rigid inter-coder-reliability frame. But a codebook approach can still help for smaller, bounded annotation tasks on job postings.

### How it could help our research

This is the right method for:

- ghost requirements
- JD authorship and screening
- anticipatory restructuring
- how seniors and juniors narrate the ladder

It is not the right method for measuring the prevalence of AI requirements across 100,000 postings.

### Recommendation

Use reflexive thematic analysis for interviews, with analytic memos after each interview and explicit comparison across cohorts. Do not hand the whole qualitative component to an automated pipeline.

## 4. LLM-Assisted Annotation and Coding

### What it is

This is the use of large language models to help label, summarize, code, or extract patterns from text.

The recent literature suggests a useful but limited role:

- LLMs can speed up first-pass coding or deductive classification
- they can often complement humans
- they still make substantive errors, flatten nuance, and drift if prompts are underspecified

The best evidence supports a **human-in-the-loop** model, not full automation.

### Where it can help

- binary or low-cardinality tagging once categories are well-defined
- extracting years-of-experience mentions
- flagging whether a posting mentions AI tools, system design, or mentorship
- proposing candidate codes for human review
- summarizing clusters of near-duplicate excerpts for analysts

### Where it should not be trusted blindly

- generating final interview themes
- deciding whether a requirement is genuinely "ghost" without external context
- replacing close reading on edge cases
- coding nuanced tone, irony, or organizational meaning without validation

### Best practice for this project

Use LLMs only after:

1. you have a human-designed codebook
2. you have a manually labeled pilot set
3. you have measured agreement against human labels

Recommended workflow:

- manually annotate a stratified pilot set
- write a codebook with inclusion / exclusion rules
- prompt the LLM to classify a held-out validation set
- inspect false positives and false negatives
- use the model only for categories where error is acceptable

### Recommendation

Treat LLMs as annotation assistants, not as qualitative researchers. They are most useful for scaling narrow coding tasks across the posting corpus after human category design.

## What These Methods Mean for Our Project

The right division of labor is:

### For exploratory hypothesis generation

- Fightin' Words
- BERTopic or STM
- descriptive plots

### For core measurement

- manually designed dictionaries
- validated extraction rules
- limited LLM-assisted scaling after validation

### For qualitative mechanism evidence

- reflexive thematic analysis
- data-prompted elicitation
- cross-cohort comparison

## Recommended Method Stack for the First Paper

If the goal is the strongest first submission, the sequence should be:

1. Run descriptive analyses of junior share, seniority composition, and requirement prevalence.
2. Use Fightin' Words and topic models for hypothesis generation only.
3. Define the core constructs carefully before any scaling.
4. Scale narrow annotation tasks with LLM assistance only where validation is good enough.
5. Run interviews and analyze them with reflexive thematic analysis.

That sequence respects what each method is actually good at.

## Bottom Line

The March 6 notes pointed to a strong methodology stack, but the methods should not carry equal weight.

For this project:

- `Fightin' Words` is a high-value early text-comparison tool.
- `Topic models` are useful for exploration and robustness, not headline identification.
- `LLM-assisted annotation` is useful only after human category design and validation.
- `Reflexive thematic analysis` is the right approach for the interview study.

## Key Sources

- Blei, Ng, and Jordan (2003), *Latent Dirichlet Allocation*: https://jmlr.org/papers/v3/blei03a.html
- Roberts et al. (2014), *Structural Topic Models for Open-Ended Survey Responses*: https://dtingley.scholars.harvard.edu/ons/topic-models-open-ended-survey-responses-applications-experiments
- Grootendorst (2022), *BERTopic*: https://arxiv.org/abs/2203.05794
- Monroe, Colaresi, and Quinn (2008), *Fightin' Words*: https://www.cambridge.org/core/journals/political-analysis/article/fightin-words-lexical-feature-selection-and-evaluation-for-identifying-the-content-of-political-conflict/81B3703230D21620B81EB6E2266C7A66
- Braun and Clarke (2006), *Using Thematic Analysis in Psychology*: https://uwe-repository.worktribe.com/output/1043060/using-thematic-analysis-in-psychology
- Braun and Clarke (2019), *Reflecting on Reflexive Thematic Analysis*: https://doi.org/10.1080/2159676X.2019.1628806
- Tai et al. (2024), *An Examination of the Use of Large Language Models to Aid Analysis of Textual Data*: https://doi.org/10.1177/16094069241231168
- Han et al. (2025), *Can large language models be used to code text for thematic analysis?*: https://doi.org/10.1007/s44163-025-00441-3
- Acemoglu et al. (2022), *Artificial Intelligence and Jobs: Evidence from Online Vacancies*: https://www.nber.org/papers/w28257
