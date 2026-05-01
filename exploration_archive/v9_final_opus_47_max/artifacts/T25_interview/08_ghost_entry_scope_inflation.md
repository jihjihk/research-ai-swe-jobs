# Artifact 8 — Ghost entry exemplars: senior-scope language in entry-level JDs

**Source:** T22 ghost forensics + T33 narrative sample. `exploration/tables/T22/top20_ghost_entry.csv`. Ghost indicators parquet: `exploration/tables/T22/ghost_indicators.parquet`.

This artifact probes the mechanism behind T22's finding that 50.3% of near-AI-token windows in 2026 postings contain hedging language ("preferred", "nice-to-have", "a plus"), vs 28.2% of far-from-AI text. AI asks cluster in wish-list sections, not hard-requirement sections.

---

## Top ghost-entry exemplars (n=5 selected from T22 top-20)

### Exemplar 1 — Visa "Software Engineer, New College Grad — 2026" (4 near-identical postings across cities)

**Kitchen-sink score: 126** (21 tech_count × 6 scope_kitchen_count)
**LLM ghost assessment: inflated**
**Seniority label:** entry (via seniority_rule keyword "new college grad")
**YOE LLM:** 0

Asks include:
- LLM fine-tuning familiarity
- GPT integration
- MLOps exposure
- Distributed systems knowledge
- Cloud architecture (AWS/GCP/Azure)
- 21 technologies listed

**Contradiction:** posting is labeled entry/new-grad, asks for LLM fine-tuning and MLOps — skills that typically require 2-3+ years of specialization. Visa posted 4 near-identical copies across San Francisco, Austin, Foster City, and other cities.

### Exemplar 2 — PNNL (Pacific Northwest National Laboratory) "Senior Software Engineer III"

**Kitchen-sink score: 324** (36 tech_count × 9 scope_kitchen_count) — highest in top-20
**LLM ghost assessment: realistic** (contrast with exemplar 1)
**Seniority label:** mid-senior
**YOE LLM:** 1 (but yoe_rule_max=16 — likely misparsed)

This posting has MS/PhD paths, clearance requirements, and 36 listed technologies. The "ghost-likeness" in this case is the extreme breadth, not entry-labeled inflation. It does NOT fit the "entry-level asking for senior scope" pattern; it fits the "senior role asking for everything" pattern.

### Exemplar 3 — DataVisor "Software Engineer, Artificial Intelligence"

**Kitchen-sink score: 165** (15 × 11)
**Seniority label:** unknown
**YOE LLM:** 2 (J3)

Asks include:
- Architect production-grade distributed services
- RAG workflows and LLM pipelines
- Fraud-detection ML pipelines (supervised / unsupervised)
- High-throughput event-driven systems (Kafka, Flink)
- Multi-agent workflow architectures

**Contradiction:** The title is "Software Engineer, Artificial Intelligence" (no explicit seniority); YOE labeled at 2 years; scope is staff/principal-level (architect production systems, own multi-agent frameworks).

### Exemplar 4 — Intellectt Inc "Fullstack .NET Architect"

**Kitchen-sink score: 170**
**LLM ghost assessment: inflated**
**Seniority label:** mid-senior
**YOE LLM:** 1

Asks for architect-level Azure cloud deployment, multi-tier distributed systems, WCF, SOA, 17 technologies. YOE=1 but asks architect-level scope. This is a "title-inflation" ghost (architect title with thin YOE ask) that T36 catalogs as typical of 2024 consultant-staffing postings; still present in 2026.

### Exemplar 5 — PayPal "Software Engineer, Routing Platform"

**Kitchen-sink score: 81** (9 × 9)
**Seniority label:** entry
**YOE LLM:** 1 (but yoe_rule_max=25; rule picked up "25 years" elsewhere in description — misparse)

Asks include:
- Migrate existing Golang/Java routing services to next-generation Kubernetes-native
- Zero-downtime transitions
- Service mesh (Istio, kgateway)
- On-call rotations
- JWT/OAuth/OIDC/mTLS zero-trust principles

**Contradiction:** labeled entry; asks mid-senior-caliber routing-platform migration experience. On-call rotation is often a senior responsibility.

---

## Aggregate pattern

T22 aggregate findings:
- **J3 (yoe≤2) YOE-scope mismatch rose 2.9% → 7.5% (2.6×).** About 7% of entry-level postings ask for ≥3 senior-scope terms — real but minority phenomenon.
- **Kitchen-sink score rose 2.6× at aggregate** (6.72 → 17.36 mean; median 0 → 8). Every 2026 posting carries some scope×tech breadth.
- **Ghost-LLM inflated rate is FLAT cross-period** (6.1% 2024 → 5.3% 2026). LLM ghost-assessment didn't rise; what rose is measurable breadth.

**Most important:** the "senior-scope in entry" phenomenon is visibly **AI-driven**. Top-20 ghost-entry postings predominantly ask for LLM / GPT / RAG / MLOps skills in new-grad / entry labeled roles.

---

## Interview questions

### To hiring managers at firms posting entry-level AI roles

1. **"Here are real entry-level SWE postings from your sector asking for LLM fine-tuning, GPT integration, distributed systems architect-level skills. (Show exemplars 1, 3, 5.) Are these genuine asks for entry-level candidates, or are they wish-list items designed to attract stronger candidates who won't meet every bullet?"**

2. **"T22's data shows 50% of near-AI-token text in postings sits next to hedging language ('preferred', 'nice-to-have', 'a plus'), vs 28% of non-AI text. Is AI being written as a soft-preference rather than a hard-requirement? Why the asymmetry?"**

3. **"Is your firm writing entry JDs with a specific candidate profile in mind, or are you writing them to attract a range of profiles that you'll narrow later in the interview? Does the 'ghost-like' appearance of the JD reflect a gap between your hiring ideal and your filter reality?"**

### To recruiters

4. **"Visa posted 4 near-identical 'New College Grad' postings across 4 cities, each asking for LLM fine-tuning and MLOps. Is this a templated job-family approach (same template, many locations) or a genuine multi-role hiring spree? How do you decide what to include in a new-grad JD vs. an associate JD?"**

5. **"When you write an entry-level posting that asks for architect-level or senior-level skills, do you expect 0%, 10%, or 50% of applicants to have those skills? What's the actual applicant-pool composition vs. the JD ask?"**

### To recent graduates / early-career candidates

6. **"Here's an entry-level SWE posting asking for LLM fine-tuning, GPT integration, and multi-agent architecture experience. (Show exemplar 1 or 3.) How does this make you feel as a candidate? Do you still apply? Does the list change your preparation strategy?"**

7. **"T22's data shows that about 7% of entry-level JDs have 3+ senior-scope terms. Did you encounter JDs that seemed mismatched with their 'entry-level' label? What did you do?"**

---

## Mechanistic readings to probe

Interviews should disambiguate between three competing mechanisms:

- **(a) Wish-list recruitment.** JDs are written optimistically; the hiring team expects to see candidates meeting 50-70% of the bullets and filters down. AI skills are aspirational signals, not filters.
- **(b) Cross-level consolidation.** 2026 entry roles genuinely require more skills than 2024 entry roles because the work genuinely shifted — bar raised.
- **(c) LLM-authored JDs.** Recruiter-adopted LLM tooling generates broader JDs mechanically (more categories cataloged). The scope increase is authorship-mediated, not demand-mediated.

T29 partially rejects (c) for content-deltas in general (scope and AI-binary persist on low-LLM subset), but doesn't specifically test entry-level posting authorship.

## Caveats

- **Ghost-LLM inflation rate is FLAT cross-period.** The LLM-adjudicated ghost rate did NOT rise. Interviewees may interpret "rising kitchen-sink score" as "more ghost" without realizing the LLM ghost-adjudication distinguishes breadth from inflation.
- **Template saturation is RARE** in this corpus (only 1 firm, Syms Strategic Group, at cosine > 0.89). Most firms post diverse JDs per cohort. The ghost pattern is not a "copy-paste-everywhere" artifact.
- **Visa's 4 new-grad postings are a single-firm pattern.** Do not generalize to all firms without the aggregate T22 data (7.5% J3 mismatch rate).
