# Artifact 9 — Requirements-section contraction without hiring-bar lowering

**Source:** T33 hiring-bar regression + narrative 50-sample. `exploration/tables/T33/narrative_sample_50.csv`.

This artifact probes the mechanism behind the REJECTED hiring-bar-lowering hypothesis (H_B). T33 tested whether requirements-section contraction in 2026 SWE postings correlates with lower YOE / credential / tech / education asks. It does NOT.

---

## Aggregate findings

### The two-classifier direction flip

Two reasonable section classifiers give OPPOSITE signs on the 2026 period effect:

| Classifier | 2024 req_share | 2026 req_share | Δ |
|---|---|---|---|
| T13 structured classifier | 23.8% | 21.3% | **−2.5 pp** |
| Simple-regex classifier | 22.4% | 27.7% | **+5.3 pp** |

Both p<1e-13 under HC3 robust SEs on n=67,962. Covariate adjustment (seniority + archetype + aggregator + log description length) preserves the direction flip: T13 period coef −0.019; simple-regex +0.030.

**Classifier choice flips the sign.** Neither classifier is ground truth; both are reasonable operationalizations of "what fraction of the posting is requirements text."

### Within-2024 SNR

- T13 classifier: within-2024 share gap 0.026, cross-period gap 0.025. **SNR 0.97** — at noise floor.
- Simple regex: within-2024 share gap 0.050, cross-period gap 0.053. **SNR 1.05** — at noise floor.

The aggregate requirements-share shift is classifier-dependent AND at within-2024 calibration noise. Cannot be cited as a substantive finding.

### Hiring-bar proxy correlations (T33 §3)

Within 2026 scraped subset, Spearman correlations of req_section_share with hiring-bar proxies:

| Proxy | T13 | Simple regex |
|---|---|---|
| YOE (LLM) | +0.05 | −0.08 |
| Credential stack depth | +0.07 | +0.13 |
| Tech count | +0.08 | −0.01 |
| Education level | +0.03 | +0.19 |

All |ρ| ≤ 0.28; direction-inconsistent across classifiers for YOE and tech_count. **The hiring-bar-lowering hypothesis predicts uniform positive correlations (more reqs → higher bar). The data shows small, inconsistent correlations.**

### The most interpretable pattern: req_chars × desc_length

Across both classifiers, Δ(req_CHARS) — absolute requirements length — is positively correlated with Δ(description length) at r≈+0.35. **Companies whose descriptions grew ALSO grew their requirements sections in absolute chars.** The SHARE declined because OTHER sections (responsibilities, benefits, about, legal) grew faster.

---

## T33 narrative 50-sample classification

50 largest-contraction postings (relative to same company × title 2024 mean) classified heuristically:

| Class | n | share | Rule |
|---|---|---|---|
| (a) Tech-req migration to responsibilities (≥5 tech keywords) | 28 | **56%** | Technical content is present but attributed to responsibilities/summary, not requirements |
| (b) Culture/benefits expansion (≥4 benefit keywords, ≤3 tech) | 15 | **30%** | Heavy perks/benefits/about-company text dilutes requirements share |
| (c) Substantive requirement loosening ("no degree required", "self-taught OK", "0+ years") | **0** | **0%** | ZERO explicit loosening language in the 50-sample contraction tail |
| (d) Other | 7 | 14% |  |

**Key result: 0 of 50 postings contain explicit requirement-loosening language.**

---

## Exemplar vignettes (paraphrased from T33 narrative_sample_50.csv)

### Exemplar A — Tech-req migration to responsibilities

A 2026 Capgemini "Software Engineer" posting has a short requirements section ("3+ years of experience, Bachelor's preferred") and a long responsibilities section detailing "Build and maintain distributed microservices, design CI/CD pipelines, integrate LLM-assisted development tools, own production incidents..." T13 attributes the technical asks to responsibilities; simple-regex attributes them to requirements.

**What it isn't:** a hiring-bar-lowered posting. The technical content is present; it's narratively located differently.

### Exemplar B — Culture/benefits expansion

A 2026 Aggregator posting features a short requirements section and a 1,500-char benefits block covering healthcare, 401k match, equity, parental leave, mental-health support, remote-work flexibility, and EEOC compliance text. The requirements section is bounded and specific; the boilerplate dilutes its share of the total.

**What it isn't:** a hiring-bar-lowered posting. The requirements are normal; the non-requirements content expanded.

### Exemplar C — No posting in the 50-sample matches this pattern

No posting in the T33 narrative 50 contains phrasing like:
- "No degree required"
- "Self-taught candidates welcome"
- "0+ years of experience"
- "Willing to train"
- "No specific tech stack required"

---

## Interview questions

### To hiring managers

1. **"T33 finds that 2026 postings with shrunk requirements sections did NOT lower their explicit hiring bars — YOE asks, credential asks, tech-count asks, and education asks did not drop. What changed about how your firm writes requirements sections?"**

2. **"56% of large-contraction postings had tech content MIGRATED to responsibilities sections (rather than removed). 30% had benefits/culture EXPANSION without corresponding content change. Does this match your firm's JD-writing practice? Did a style-guide or JD template change between 2024 and 2026?"**

3. **"If you use LLM-based JD drafting tools, do those tools produce more balanced 'responsibilities + benefits + qualifications' structure than hand-written JDs? Could the 2026 section-share shift be partially a template artifact?"**

### To recruiters / talent-acquisition partners

4. **"T33 found ZERO out of 50 largest-contraction postings contain explicit requirement-loosening language (no 'self-taught', no 'no degree required', no '0+ years'). If your firm wanted to genuinely lower the hiring bar, how would you signal that in a JD? Would you use explicit language, or silent reduction?"**

5. **"The data shows requirements-section SHARE dropped while requirements-section CHARS stayed stable (r=+0.35 with description length growth). This is 'narrative expansion' rather than 'requirements contraction'. Does this match how you write JDs today vs. 2 years ago?"**

### To candidates

6. **"When you read a modern SWE JD, where do you look for the hiring bar? Requirements section? Responsibilities section? Company size/stage? The data suggests the requirements section is NOT a reliable thermometer of hiring bar."**

7. **"Between 2024 and 2026, did your perception of hiring standards change? Did JDs seem more or less selective than before? Can you reconcile a perception of harder hiring standards with the finding that explicit requirements didn't rise?"**

### To JD-writers / content writers

8. **"The classifier flip is the substantive finding: two reasonable classifiers disagree on the direction of the requirements-share shift. One interprets markdown-bold headers as requirements; the other uses phrase catalogs. Do you have a preferred convention for where technical asks should live (requirements vs. responsibilities vs. summary)?"**

---

## Mechanistic readings to probe

Interviews should disambiguate:

- **(a) Narrative-expansion dominance.** The non-requirements sections (responsibilities, benefits, about-company, legal) expanded faster than requirements did. Requirements became a smaller share of a bigger posting, without shrinking in absolute terms. Supported by: r=+0.35 Δreq_chars × Δdesc_len; benefits +89%, responsibilities +49% per T13.

- **(b) LLM-authoring template effect.** Recruiter-adopted LLM JD-drafting tools produce more boilerplate sections (benefits, legal, about-company) without necessarily changing requirements. T29 confirmed length growth is 52% LLM-mediated; content deltas are only 12% LLM-mediated for AI-binary.

- **(c) Compliance/legal expansion.** Pay transparency laws (California 2023, NY 2023, others) require compensation disclosures; EEO/DEI statements expanded. This adds chars to benefits/legal sections mechanically.

T38 independently ruled out the hiring-selectivity mechanism (volume-down firms should have MORE requirements; they have slightly less). H_B (T33 hiring-bar lowering via requirements shrink) is independently rejected.

## Caveats

- **The T13 classifier and simple-regex classifier are both heuristic.** Neither is ground-truth. An LLM-adjudicated section-classifier on a 50-row sample might resolve the flip at analysis-phase.
- **The narrative 50-sample is regex-heuristic classification**, not LLM adjudication. The 0% loosening is robust because the regex looks for explicit phrasing; subtle loosening ("degree or equivalent experience") would slip through.
- **The exploration did NOT test whether actual hiring selectivity changed** (that requires applicant-pool / hiring-rate data). It only tested whether JD-side proxies (YOE, credentials, tech count, education) changed. Interview protocol should surface this distinction: "what employers ASK for" vs "who employers HIRE" are different questions.
